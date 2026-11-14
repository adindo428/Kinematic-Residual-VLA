// Rebuild a snapshot workbook from the canonical CSV/JSON ledger, not vice versa.
import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

const tracker=process.argv[2];
const output=process.argv[3];
if(!tracker || !output) throw new Error('Expected tracker directory and output directory');
const data=JSON.parse(await fs.readFile(path.join(tracker,'snapshot.json'),'utf8'));
await fs.mkdir(output,{recursive:true});
const wb=Workbook.create();
const names=['Collection','Sessions','Attempts','Training runs','Evaluation trials','Evaluation results'];
const sheets=Object.fromEntries(names.map(name=>[name,wb.worksheets.add(name)]));
const col=i=>{let s='';for(i++;i;i=Math.floor((i-1)/26))s=String.fromCharCode(65+(i-1)%26)+s;return s;};
function grid(name,headers,rows,subtitle){
  const s=sheets[name],end=col(headers.length-1),last=Math.max(6,5+rows.length);
  s.showGridLines=false;
  s.getRange(`A1:${end}${last}`).format.font={name:'Arial',size:10,color:'#253247'};
  s.getRange(`A1:${end}${last}`).format.rowHeight=23;
  s.getRange(`A1:${end}${last}`).format.verticalAlignment='center';
  s.getRange('A2').values=[[name==='Collection'?'Kinematic Residual VLA collection':name]];
  s.getRange('A2').format.font={name:'Arial',size:15,bold:true,color:'#172C49'};
  s.getRange('A3').values=[[subtitle]];
  s.getRange('A3').format.font={name:'Arial',size:10,italic:true,color:'#596579'};
  s.getRange(`A5:${end}5`).values=[headers];
  s.getRange(`A5:${end}5`).format={fill:'#243C5A',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:36,wrapText:true,horizontalAlignment:'center',verticalAlignment:'center'};
  if(rows.length)s.getRange(`A6:${end}${5+rows.length}`).values=rows.map(row=>row.map(v=>v??null));
  else s.getRange('A6').values=[['No records yet']];
  s.getRange(`A1:${end}${last}`).format.columnWidth=19;
  s.getRange(`A1:A${last}`).format.columnWidth=29;
  if(name!=='Collection')s.freezePanes.freezeRows(5);
  s.tabColor=name==='Collection'?'#243C5A':'#A9BDD3';
  return s;
}
const collection=grid('Collection',['Object','Split','Target','DOM exported','Study included','Remaining','Progress'],
  data.progress.map(r=>[r.task_id.replaceAll('_',' '),r.split,r.target,r.exported,r.included,null,null]),
  `Updated ${data.generated_utc.slice(0,16).replace('T',' ')} UTC. Inclusion requires DOM export and speed review.`);
for(let i=6;i<6+data.progress.length;i++){
  collection.getRange(`F${i}:G${i}`).formulas=[[`=MAX(0,C${i}-E${i})`,`=IF(C${i}=0,"n.a.",E${i}/C${i})`]];
}
collection.getRange('G6:G20').setNumberFormat('0.0%');
collection.getRange('A22:B22').values=[['Total target',null]];
collection.getRange('B22').formulas=[['=SUM(C6:C20)']];
collection.getRange('D22:E22').values=[['Study included',null]];
collection.getRange('E22').formulas=[['=SUM(E6:E20)']];
collection.getRange('A24').values=[['Per block: 2 stationary, 8 slow, 8 medium, 2 brisk. Eight train blocks, one validation, one test per object.']];
collection.getRange('A26').values=[['Operator controls: N begin; V success; X failure; Q finish. Camera recovery requires a new N.']];
collection.getRange('A28').values=[['Workbook is a snapshot. Edit reviews.csv, training_runs.csv and evaluation_trials.csv; refresh before paper export.']];
collection.getRange('A30').values=[['No quota credit is assigned to pending reviews or pilots. Speed bins: <1, 1–4, 4–8, 8–12 cm/s.']];
const sessions=grid('Sessions',['Session','Started UTC','Object','Split','Attempts','Successes','Failed','Incomplete','Exported','Included','Error','Raw recording'],
 data.sessions.map(r=>[r.session,new Date(r.created_utc),r.task_id,r.split,r.attempts,r.operator_successes,r.failed,r.incomplete,r.exported,r.included,r.error,r.raw_path]),
 'Source: local collection manifests, attempt ledgers and recorder summaries.');
sessions.getRange('A1:A20').format.columnWidth=55;
sessions.getRange('B6:B100').setNumberFormat('yyyy-mm-dd hh:mm');
sessions.getRange('K1:L20').format.columnWidth=60;
const attempts=grid('Attempts',['Attempt ID','Session','Object','Split','Outcome','Exported','Included','Review','Speed bin','Target ID','Reason','Seconds','Raw episode','Dataset'],
 data.attempts.map(r=>[r.attempt_id,r.session,r.task_id,r.split,r.outcome,r.exported,r.included,r.decision,r.speed_bin,r.target_id,r.reason,r.duration_s,r.raw_episode,r.dataset_path]),
 'Source: raw episode files and completed DOM exports. Reviews stay keyed to immutable attempt IDs.');
attempts.getRange('A1:A20').format.columnWidth=32;attempts.getRange('B1:B20').format.columnWidth=55;
attempts.getRange('L6:L100').setNumberFormat('0.0');attempts.getRange('M1:N20').format.columnWidth=70;
const runs=grid('Training runs',['Run ID','Model','Variant','Dataset version','Checkpoint','Seed','Status','Updates','Learning rate','Train loss','Val. action MSE','Config','Notes'],
 data.runs.map(r=>[r.run_id,r.model,r.variant,r.dataset_version,r.checkpoint,r.seed===''?null:Number(r.seed),r.status,r.updates===''?null:Number(r.updates),r.learning_rate===''?null:Number(r.learning_rate),r.train_loss===''?null:Number(r.train_loss),r.validation_action_mse===''?null:Number(r.validation_action_mse),r.config_path,r.notes]),
 'Source: training_runs.csv. No training measurements have been inferred from the collection plan.');
const trials=grid('Evaluation trials',['Trial ID','Run ID','Object','Condition','Evaluation set','Outcome','Seconds','Failure reason','Video','Recorded UTC','Notes'],
 data.trials.map(r=>[r.trial_id,r.run_id,r.task_id,r.condition,r.evaluation_set,r.outcome,r.duration_s===''?null:Number(r.duration_s),r.failure_reason,r.video_path,r.recorded_utc?new Date(r.recorded_utc):null,r.notes]),
 'Source: evaluation_trials.csv. Takeover, timeout and aborted trials count as failures; pending trials are excluded.');
const results=grid('Evaluation results',['Run ID','Object','Condition','Evaluation set','Successes','Trials','Success rate','95% lower','95% upper'],
 data.evaluation.map(r=>[r.run_id,r.task_id,r.condition,r.evaluation_set,r.successes,r.trials,null,r.ci_low,r.ci_high]),
 'Nominal 95% Wilson binomial intervals; development and test remain separate. No trials means unavailable, not 0%.');
for(let i=6;i<6+data.evaluation.length;i++) results.getRange(`G${i}`).formulas=[[`=IF(F${i}=0,"n.a.",E${i}/F${i})`]];
results.getRange('G6:I100').setNumberFormat('0.0%');
wb.recalculate();
if(collection.getRange('B22').values[0][0]!==1000) throw new Error('Collection target reconciliation failed');
const included=data.progress.reduce((s,r)=>s+r.included,0);
if(collection.getRange('E22').values[0][0]!==included) throw new Error('Inclusion reconciliation failed');
// Check an input change, then restore before export.
const original=collection.getRange('E6').values;
collection.getRange('E6').values=[[1]];
wb.recalculate();
if(collection.getRange('F6').values[0][0]!==data.progress[0].target-1) throw new Error('Remaining quota did not recalculate');
collection.getRange('E6').values=original;
wb.recalculate();
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!',options:{useRegex:true,maxResults:20},maxChars:2000});
await fs.writeFile(path.join(output,'formula_check.ndjson'),errors.ndjson);
for(const name of names){
 const preview=await wb.render({sheetName:name,range:name==='Collection'?'A1:G31':`A1:${name==='Attempts'?'L':name==='Sessions'?'K':'I'}9`,scale:1,format:'png'});
 await fs.writeFile(path.join(output,name.replaceAll(' ','_')+'.png'),new Uint8Array(await preview.arrayBuffer()));
}
const xlsx=await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(path.join(output,'Kinematic_Residual_VLA_Tracker.xlsx'));
console.log(JSON.stringify({output:path.join(output,'Kinematic_Residual_VLA_Tracker.xlsx'),sheets:names,included}));
