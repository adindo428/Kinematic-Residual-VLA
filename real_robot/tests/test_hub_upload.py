import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import shutil
from unittest.mock import Mock,patch

from real_robot.hub_upload import upload_dataset,verify_upload_access,find_published_base,reconcile_append


def remote_file(name,payload):
    return SimpleNamespace(path=name,size=len(payload),lfs=None,
        blob_id=hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest())


class HubUploadTests(unittest.TestCase):
    def fixture(self,root):
        local=root/'dom';old=root/'remote'
        for folder,n in [(old,1),(local,2)]:
            (folder/'meta').mkdir(parents=True)
            info=dict(total_episodes=n,total_frames=n*10,features={},fps=25,robot_type='aloha_solo',
                      data_path='data/{episode_index}',video_path='videos/{episode_index}',chunks_size=1000)
            (folder/'meta/info.json').write_text(json.dumps(info))
            for name in ['episodes','episodes_stats','sources','camera']:
                (folder/f'meta/{name}.jsonl').write_text(''.join(json.dumps({'episode_index':i})+'\n' for i in range(n)))
            (folder/'meta/tasks.jsonl').write_text('{"task_index":0}\n')
            (folder/'meta/splits.json').write_text(json.dumps({'train':list(range(n))}))
            (folder/'meta/raw_hashes.json').write_text(json.dumps({str(i):str(i) for i in range(n)}))
        # A prepared card is preserved without relying on image-schema details in this fixture.
        (local/'README.md').write_text('2 episodes')
        (local/'data').mkdir();(local/'data/episode_000000.parquet').write_bytes(b'original bytes')
        (local/'data/episode_000001.parquet').write_bytes(b'new bytes')
        remote=[remote_file('meta/info.json',(old/'meta/info.json').read_bytes()),
                remote_file('data/episode_000000.parquet',b'original bytes'),remote_file('.gitattributes',b'attributes')]
        after=[remote_file('meta/info.json',(local/'meta/info.json').read_bytes()),*remote[1:]]
        after.append(remote_file('data/episode_000001.parquet',b'new bytes'))
        return local,old,remote,after

    def test_append_adds_new_files_and_never_deletes_or_uploads_existing_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            local,old,remote,after=self.fixture(Path(directory))
            api=Mock();api.repo_info.return_value=Mock(private=False,sha='previous')
            api.list_repo_tree.side_effect=[remote,after];api.create_commit.return_value=Mock(oid='updated')
            def download(repo,name,**kwargs):return str((local if kwargs['revision']=='updated' else old)/name)
            with patch('real_robot.hub_upload.authenticated_repo',return_value=(api,'owner/data')), \
                 patch('real_robot.hub_upload.verify_upload_access',return_value=api.repo_info.return_value), \
                 patch('real_robot.dom_dataset.validate_dataset',return_value=dict(episodes=2,frames=20)), \
                 patch('huggingface_hub.hf_hub_download',side_effect=download):
                result=upload_dataset(local,'owner/data',public=True,append=True,expected_parent='older_recording_base')
            api.create_branch.assert_not_called();api.upload_folder.assert_not_called()
            self.assertEqual(api.create_commit.call_args.kwargs['parent_commit'],'previous')
            operations=api.create_commit.call_args.kwargs['operations']
            self.assertTrue(all(type(op).__name__=='CommitOperationAdd' for op in operations))
            names={op.path_in_repo for op in operations}
            self.assertNotIn('data/episode_000000.parquet',names)
            self.assertIn('data/episode_000001.parquet',names)
            self.assertEqual(result['new_episodes'],1)
            self.assertEqual(result['preserved_episode_files'],1)

    def test_existing_episode_modification_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as directory:
            local,old,remote,after=self.fixture(Path(directory))
            (local/'data/episode_000000.parquet').write_bytes(b'changed')
            api=Mock();api.repo_info.return_value=Mock(private=False,sha='previous');api.list_repo_tree.return_value=remote
            with patch('real_robot.hub_upload.authenticated_repo',return_value=(api,'owner/data')), \
                 patch('real_robot.hub_upload.verify_upload_access',return_value=api.repo_info.return_value), \
                 patch('real_robot.dom_dataset.validate_dataset',return_value=dict(episodes=2,frames=20)), \
                 patch('huggingface_hub.hf_hub_download',side_effect=lambda repo,name,**kw:str(old/name)):
                with self.assertRaisesRegex(ValueError,'existing episode file'):
                    upload_dataset(local,'owner/data',public=True,append=True)
            api.create_commit.assert_not_called()

    def test_replacement_disabled(self):
        with self.assertRaisesRegex(ValueError,'Replacement is disabled'):
            upload_dataset('/unused','owner/data',replace_existing=True)

    def test_stale_cumulative_export_is_reconciled_instead_of_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            local,old,remote,_=self.fixture(Path(directory))
            (local/'meta/sources.jsonl').write_text('{"episode_index":0,"source_episode":"different"}\n')
            api=Mock();info=Mock(private=False,sha='new_head');api.list_repo_tree.return_value=remote
            with patch('real_robot.hub_upload.authenticated_repo',return_value=(api,'owner/data')), \
                 patch('real_robot.hub_upload.verify_upload_access',return_value=info), \
                 patch('real_robot.dom_dataset.validate_dataset',return_value=dict(episodes=2,frames=20)), \
                 patch('huggingface_hub.hf_hub_download',side_effect=lambda repo,name,**kw:str(old/name)), \
                 patch('real_robot.hub_upload.reconcile_append',return_value={'new_episodes':1}) as reconcile:
                self.assertEqual(upload_dataset(local,'owner/data',public=True,append=True,expected_parent='old_head'),{'new_episodes':1})
            reconcile.assert_called_once_with(local.resolve(),'owner/data',api,info,True,0)
            api.create_commit.assert_not_called()

    def test_parent_commit_conflict_refreshes_and_retries_without_force(self):
        for status in (409,412):
            with self.subTest(status=status):self.check_parent_commit_conflict(status)

    def check_parent_commit_conflict(self,status):
        from huggingface_hub.errors import HfHubHTTPError
        from requests import Response
        response=Response();response.status_code=status
        with tempfile.TemporaryDirectory() as directory:
            local,old,remote,after=self.fixture(Path(directory))
            api=Mock();api.list_repo_tree.side_effect=[remote,remote,after]
            api.create_commit.side_effect=[HfHubHTTPError('parent commit changed',response=response),Mock(oid='updated')]
            def download(repo,name,**kw):return str((local if kw['revision']=='updated' else old)/name)
            with patch('real_robot.hub_upload.authenticated_repo',return_value=(api,'owner/data')), \
                 patch('real_robot.hub_upload.verify_upload_access',side_effect=[Mock(private=False,sha='previous'),Mock(private=False,sha='fresh')]), \
                 patch('real_robot.dom_dataset.validate_dataset',return_value=dict(episodes=2,frames=20)), \
                 patch('huggingface_hub.hf_hub_download',side_effect=download):
                result=upload_dataset(local,'owner/data',public=True,append=True)
            self.assertEqual(result['new_episodes'],1)
            self.assertEqual([c.kwargs['parent_commit'] for c in api.create_commit.call_args_list],['previous','fresh'])

    def test_retry_of_already_published_sources_creates_no_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            local,old,remote,_=self.fixture(Path(directory))
            (old/'data').mkdir();shutil.copyfile(local/'data/episode_000000.parquet',old/'data/episode_000000.parquet')
            api=Mock();api.list_repo_tree.return_value=remote;api.repo_info.return_value=Mock(sha='published')
            with patch('real_robot.hub_upload.find_published_base',return_value=old), \
                 patch('real_robot.merge_datasets.merge_datasets',return_value={'episodes':1,'frames':10}):
                result=reconcile_append(local,'owner/data',api,Mock(sha='published'),True,0)
            self.assertEqual(result['mode'],'already_uploaded')
            self.assertEqual(result['new_episodes'],0)
            self.assertEqual(result['preserved_episode_files'],1)
            api.create_commit.assert_not_called()

    def test_permission_probe_requests_fresh_lfs_object_without_commit(self):
        api=Mock(token='sentinel',endpoint='https://huggingface.co')
        api.repo_info.return_value=Mock(private=False,sha='previous')
        with patch('huggingface_hub.lfs.post_lfs_batch_info',return_value=([{}],[])) as batch:
            verify_upload_access(api,'owner/data',public=True)
            first=batch.call_args.args[0][0]
            verify_upload_access(api,'owner/data',public=True)
            second=batch.call_args.args[0][0]
        self.assertNotEqual(first.sha256,second.sha256)
        self.assertEqual(hashlib.sha256(first.sample).digest(),first.sha256)
        self.assertEqual(batch.call_args.kwargs['token'],'sentinel')
        self.assertEqual(batch.call_args.kwargs['revision'],'main')
        api.create_commit.assert_not_called();api.upload_file.assert_not_called()

    def test_permission_denial_explains_environment_override(self):
        from huggingface_hub.errors import HfHubHTTPError
        from requests import Response
        response=Response();response.status_code=403
        api=Mock(token=None,endpoint='https://huggingface.co')
        api.repo_info.return_value=Mock(private=False,sha='previous')
        with patch.dict('os.environ',{'HF_TOKEN':'sentinel'}), \
             patch('huggingface_hub.lfs.post_lfs_batch_info',side_effect=HfHubHTTPError('Forbidden',response=response)):
            with self.assertRaisesRegex(PermissionError,'HF_TOKEN environment variable') as error:
                verify_upload_access(api,'owner/data',public=True)
        self.assertIn('write access to repository contents',str(error.exception))
        self.assertNotIn('sentinel',str(error.exception))
        api.create_commit.assert_not_called()

    def test_collector_refuses_bad_upload_credentials_before_opening_hardware(self):
        from real_robot.collect import main
        with patch('real_robot.hub_upload.authenticated_repo',return_value=(Mock(),'owner/data')), \
             patch('real_robot.hub_upload.verify_upload_access',side_effect=PermissionError('write permission missing')), \
             patch('real_robot.collect.LeaseOwner') as lease, \
             patch('real_robot.collect.subprocess.Popen') as process, \
             patch('sys.stderr',new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as error:
                main(['--task','striped_ball','--hub-repo','owner/data','--public'])
        self.assertEqual(error.exception.code,2)
        lease.assert_not_called();process.assert_not_called()

    def test_uploader_checks_access_before_expensive_video_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            local,_,_,_=self.fixture(Path(directory))
            with patch('real_robot.hub_upload.authenticated_repo',return_value=(Mock(),'owner/data')), \
                 patch('real_robot.hub_upload.verify_upload_access',side_effect=PermissionError('write permission missing')), \
                 patch('real_robot.dom_dataset.validate_dataset') as validate:
                with self.assertRaises(PermissionError):upload_dataset(local,'owner/data',public=True,append=True)
            validate.assert_not_called()

    def test_find_published_base_reuses_exact_local_export_at_pinned_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,old,_,_=self.fixture(root)
            stale=root/'collections/older/dom';shutil.copytree(old,stale)
            (stale/'meta/sources.jsonl').write_text('wrong source\n')
            base=root/'collections/session/dom-combined';shutil.copytree(old,base)
            api=Mock(token=None,endpoint='https://huggingface.co')
            with patch('huggingface_hub.hf_hub_download',side_effect=lambda repo,name,**kw:str(old/name)) as download, \
                 patch('huggingface_hub.snapshot_download') as snapshot:
                actual=find_published_base(api,'owner/data',Mock(sha='published'),root)
            self.assertEqual(actual,base.resolve())
            self.assertTrue(all(call.kwargs['revision']=='published' for call in download.call_args_list))
            snapshot.assert_not_called()

    def test_missing_local_base_downloads_pinned_repository_before_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,old,_,_=self.fixture(root)
            target=root/'hub_snapshots/owner--data/published'
            api=Mock(token=None,endpoint='https://huggingface.co')
            with patch('huggingface_hub.hf_hub_download',side_effect=lambda repo,name,**kw:str(old/name)), \
                 patch('huggingface_hub.snapshot_download',return_value=str(target)) as snapshot:
                self.assertEqual(find_published_base(api,'owner/data',Mock(sha='published'),root),target.resolve())
            self.assertEqual(snapshot.call_args.kwargs['revision'],'published')
            self.assertEqual(snapshot.call_args.kwargs['repo_type'],'dataset')

    def test_empty_repository_has_no_append_base(self):
        from huggingface_hub.errors import EntryNotFoundError
        with tempfile.TemporaryDirectory() as directory, \
             patch('huggingface_hub.hf_hub_download',side_effect=EntryNotFoundError('No metadata')), \
             patch('huggingface_hub.snapshot_download') as snapshot:
            self.assertIsNone(find_published_base(Mock(),'owner/data',Mock(sha='empty'),Path(directory)))
        snapshot.assert_not_called()

    def test_collection_without_append_flag_inherits_published_images_and_base(self):
        from real_robot.collect import prepare_upload
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);(base/'meta').mkdir();prep=base/'meta/image_preprocessing.json'
            prep.write_text('{}')
            args=SimpleNamespace(hub_repo='owner/data',public=True,append_to=None,image_preprocessing=None)
            with patch('real_robot.hub_upload.authenticated_repo',return_value=(Mock(),'owner/data')), \
                 patch('real_robot.hub_upload.verify_upload_access',return_value=Mock(sha='published')), \
                 patch('real_robot.hub_upload.find_published_base',return_value=base), \
                 patch('real_robot.hub_upload.verify_append_base',return_value='published'), \
                 patch('real_robot.dom_dataset.validate_dataset') as validate:
                self.assertEqual(prepare_upload(args),'published')
            self.assertEqual(args.append_to,base)
            self.assertEqual(args.image_preprocessing,prep)
            validate.assert_called_once_with(base,decode_videos=False)

    def test_collection_explicit_stale_base_is_rejected_before_recording(self):
        from real_robot.collect import prepare_upload
        args=SimpleNamespace(hub_repo='owner/data',public=True,append_to=Path('/explicit'),image_preprocessing=None)
        with patch('real_robot.hub_upload.authenticated_repo',return_value=(Mock(),'owner/data')), \
             patch('real_robot.hub_upload.verify_upload_access',return_value=Mock(sha='published')), \
             patch('real_robot.hub_upload.find_published_base') as discover, \
             patch('real_robot.dom_dataset.validate_dataset'), \
             patch('real_robot.hub_upload.verify_append_base',side_effect=ValueError('outdated base')):
            with self.assertRaisesRegex(ValueError,'outdated base'):prepare_upload(args)
        discover.assert_not_called()


if __name__=='__main__':unittest.main()
