def train_distributed(root, variant, runtime, recipe, device, resume_stopped=False):
    root = Path(root)
    manifest = load_family(root)
    epoch_control = _operational_epoch_control(root, variant, recipe, manifest)
    max_epochs = epoch_control['epochs']
    rule = recipe.get('early_stopping')
    validate_rule(rule)
    if recipe['effective_batch'] != 352 or not 1 <= max_epochs <= 100:
        raise ValueError('The study requires batch 352 and at most 100 epochs')
    seed_all()
    destination = root/'runs'/variant
    fingerprint = digest({'manifest':manifest['fingerprint'],'recipe':recipe,
                          'variant':variant,'world_size':dist.get_world_size()})
    rank = dist.get_rank()
    with distributed_lock(destination):
        split = json.loads((root/'split.json').read_text())['splits']
        data = build_dataset(root,split['train'],training=True)
        if len(data) != manifest['train_frames']:
            raise ValueError('Training dataset differs from frozen frame count')
        val_loader = None
        if rank == 0:
            val = build_dataset(root,split['validation'])
            val_loader = make_loader(val,runtime,indices=json.loads((root/'validation_anchors.json').read_text()))
        policy = build_policy(root,variant,device)
        policy.config.optimizer_lr = recipe['lr']
        policy.config.scheduler_warmup_steps = 1000
        policy.config.scheduler_decay_steps = epoch_control['scheduler_horizon_epochs']*math.ceil(len(data)/352)
        optimizer = make_optimizer(policy,recipe['lr'],runtime['fused'])
        scheduler = make_scheduler(optimizer,policy.config.scheduler_decay_steps)
        progress = {'epoch':0,'cursor':0,'updates':0,'samples':0}
        best,wandb_id,state = None,None,None
        if (destination/'latest.json').exists():
            state = load_generation(destination,policy,optimizer,scheduler,fingerprint,restore_random=False)
            progress,best,wandb_id = state['progress'],state['best'],state['wandb_id']
        elif (destination/'provenance.json').exists():
            previous = json.loads((destination/'provenance.json').read_text())
            if previous['fingerprint'] != fingerprint:
                raise ValueError('Run directory belongs to a different distributed experiment')
            wandb_id = previous['wandb_id']
        model = wrap_policy(policy,device)
        if state is not None:
            restore_distributed(state)
            del state
        else:
            # Different denoising noise per rank; augmentation stays per sample.
            torch.manual_seed(42+rank)
            torch.cuda.manual_seed(42+rank)
        if progress.get('early_stopped') and resume_stopped and progress['epoch'] < max_epochs:
            progress['early_stopped'] = False
            progress.pop('completion_reason', None)
            progress['early_stopping']['bad_validations'] = 0
            progress.setdefault('resume_stopped_epochs', []).append(progress['epoch'])
            leader_call(lambda:(destination/'training_complete.json').unlink(missing_ok=True))
        elif (progress.get('early_stopped') or progress['epoch'] >= max_epochs) and not progress.get('pending_validation'):
            return progress
        progress['operational_epoch_cap'] = epoch_control['epochs']
        progress['scheduler_horizon_epochs'] = epoch_control['scheduler_horizon_epochs']
        leader_call(lambda:atomic_json(destination/'operational-control.json',epoch_control))
        logging.info('%s total epoch cap=%d; original scheduler horizon=%d; restored updates=%d',variant,max_epochs,epoch_control['scheduler_horizon_epochs'],progress['updates'])
        writer = None
        def init_writer():
            nonlocal writer
            import wandb
            run_id = wandb_id or wandb.util.generate_id()
            writer = wandb.init(project='Kinematic-Residual-VLA',entity=os.environ.get('DVLA_WANDB_ENTITY','aparame-clemson-university'),
                name=variant,id=run_id,resume='allow',dir=str(destination),
                config={'manifest':manifest,'recipe':recipe,'variant':variant,'runtime':runtime,'world_size':dist.get_world_size(),'operational_epoch_cap':epoch_control['epochs'],'operational_training_control':epoch_control})
            atomic_json(destination/'provenance.json',{'fingerprint':fingerprint,'wandb_id':run_id,
                'variant':variant,'manifest':manifest['fingerprint'],'runtime':runtime,'recipe':recipe,
                'world_size':dist.get_world_size(),'backend':'DDP/NCCL','operational_training_control':epoch_control})
            return run_id
        wandb_id = leader_call(init_writer)
        stop = {'requested':False}
        def request_stop(signum,frame): stop['requested']=True
        handlers = {sig:signal.signal(sig,request_stop) for sig in (signal.SIGUSR1,signal.SIGTERM)}
        last_save = time.monotonic()
        def save(milestone=False, final_checkpoint=False):
            nonlocal last_save
            save_distributed(destination,policy,optimizer,scheduler,progress,fingerprint,wandb_id,best,milestone,device,final_checkpoint)
            last_save = time.monotonic()
        def finish_epoch():
            nonlocal best
            def validate():
                try:
                    return evaluate(policy,val_loader,runtime['precision'],device,should_stop=lambda:stop['requested'])
                except InterruptedError:
                    return None
            metrics = leader_call(validate)
            if metrics is None:
                stop['requested']=True
                return False
            progress['early_stopped'] = record_validation(progress,metrics,rule)
            finished = progress['early_stopped'] or progress['epoch'] >= max_epochs
            if finished:
                progress['completion_reason'] = 'early_stopping' if progress['early_stopped'] else 'epoch_cap'
            value = metrics['h20/action_mse']
            if best is None or value < best['value']:
                best = {'value':value,'epoch':progress['epoch'],'updates':progress['updates']}
            def publish():
                atomic_json(destination/f"validation-epoch-{progress['epoch']:03d}.json",metrics)
                report = {'Validation/'+key:value for key,value in metrics.items()}
                report.update({'Stop/bad_validations':progress.get('early_stopping',{}).get('bad_validations',0),
                               'Stop/early_stopped':progress['early_stopped'],'Stop/completed_epochs':progress['epoch']})
                writer.log(report,step=progress['updates'])
                logging.info('%s epoch=%d validation_mse=%.6g bad_validations=%d completion=%s',
                    variant,progress['epoch'],value,report['Stop/bad_validations'],progress.get('completion_reason','training'))
            leader_call(publish)
            progress['pending_validation']=False
            milestone = finished or progress['epoch'] in (1,5) or progress['epoch']%10==0
            save(milestone,final_checkpoint=finished)
            return True
        try:
            if not (destination/'latest.json').exists(): save()
            if progress.get('pending_validation') and not finish_epoch():
                save();return {**progress,'interrupted':True}
            optimizer.zero_grad(set_to_none=True)
            while progress['epoch'] < max_epochs and not progress.get('early_stopped'):
                loader = distributed_loader(data,runtime,progress['epoch'],progress['cursor'],recipe['sampling'])
                for record in updates(model,optimizer,scheduler,loader,progress,runtime['precision'],device):
                    if rank == 0 and (progress['updates']%10==0 or progress['updates']==1):
                        logging.info('%s ranks=%d epoch=%.6f updates=%d samples=%d loss=%.6g rate=%.2f',
                            variant,dist.get_world_size(),progress['epoch']+progress['cursor']/len(data),progress['updates'],
                            progress['samples'],record['loss'],record['samples_per_second'])
                        writer.log({'Train/loss':record['loss'],'Train/lr':optimizer.param_groups[0]['lr'],
                            'Train/samples':progress['samples'],'Train/optimizer_updates':progress['updates'],
                            'Train/epoch':progress['epoch']+progress['cursor']/len(data),
                            'Train/samples_per_second':record['samples_per_second'],'Train/update_latency_s':record['update_latency_s'],
                            'Train/world_size':dist.get_world_size()},step=progress['updates'])
                    halt = stopped(stop,device)
                    due = torch.tensor(int(time.monotonic()-last_save>=1800),device=device)
                    dist.all_reduce(due,op=dist.ReduceOp.MAX)
                    # Protect initial training before the first 30-minute save.
                    early_save = progress['updates'] <= 1000 and progress['updates'] % 25 == 0
                    if halt or bool(due) or early_save:save()
                    if halt:return {**progress,'interrupted':True}
                del loader
                if progress['cursor'] != len(data):raise RuntimeError('Distributed epoch omitted samples')
                progress['epoch'] += 1;progress['cursor']=0;progress['pending_validation']=True
                save()
                if not finish_epoch():save();return {**progress,'interrupted':True}
            leader_call(lambda:atomic_json(destination/'training_complete.json',{**progress,'fingerprint':fingerprint,'best':best}))
            return progress
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)
            if writer is not None:writer.finish(exit_code=0 if progress.get('completion_reason') and not progress.get('pending_validation') else 1)
