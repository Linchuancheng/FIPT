"""Summarize test results at the validation-selected epoch for each seed."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def summarize(root, allow_partial=False, selection_metric=None):
    rows=[]; issues=[]; expected=set()
    for log in sorted(Path(root).glob('seed_*/log.txt')):
        run=log.parent
        config=json.loads((run/'run_config.json').read_text(encoding='utf-8-sig'))
        expected.update(config.get('requested_seeds',[config['seed']]))
        if not (run/'completed.json').is_file() and not allow_partial:
            issues.append({'run':run.name,'reason':'incomplete; excluded'})
            continue
        config=json.loads((run/'run_config.json').read_text(encoding='utf-8-sig'))
        metric=selection_metric or 'balanced_accuracy'
        if config.get('selection_metric') != metric:
            issues.append({'run':run.name,'reason':'Log selection differs from the saved checkpoint criterion; checkpoint unchanged'})
        records=[json.loads(line) for line in log.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        if not records:
            issues.append({'run':run.name,'reason':'empty log'})
            continue
        epochs = [record['epoch'] for record in records]
        if len(epochs) != len(set(epochs)):
            raise ValueError(f'{log}: duplicate epochs')
        if not allow_partial and sorted(epochs) != list(range(config['args']['epochs'])):
            raise ValueError(f'{log}: completed marker does not match the epoch log')
        records.sort(key=lambda record: record['epoch'])
        values=[r.get('val_'+metric,float('nan')) for r in records]
        if not all(np.isfinite(v) for v in values):
            raise ValueError(f'{log}: non-finite/missing validation {metric}')
        # max returns the first item on a tie, matching checkpoint-best selection.
        best=max(records,key=lambda r:r['val_'+metric])
        rows.append(dict(seed=config['seed'],epoch=best['epoch'],selection_metric=metric,
                         val_selection=best['val_'+metric],
                         **{k:v for k,v in best.items() if k.startswith('test_')}))
    if not rows:
        raise ValueError('No eligible completed runs found. --allow-partial includes unfinished runs explicitly.')
    if len({r['selection_metric'] for r in rows})!=1:
        raise ValueError('Mixed checkpoint selection criteria; summarize separately or specify --selection-metric')
    metrics={}
    for key in rows[0]:
        if key.startswith('test_'):
            values=np.asarray([r[key] for r in rows],dtype=float)
            metrics[key]={'mean':float(values.mean()),'std_ddof0':float(values.std(ddof=0)),
                          'std_ddof1':float(values.std(ddof=1)) if len(values)>1 else None}
    result=dict(runs=len(rows),missing_seeds=sorted(expected-{r['seed'] for r in rows}),selection_metric=rows[0]['selection_metric'],
                selection='highest validation score; earliest logged epoch breaks ties',
                allow_partial=allow_partial,issues=issues,metrics=metrics)
    return rows,result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path,
                   default=Path(__file__).resolve().parent / 'finetuning_results' / 'mer_labram_valence',
                   help='One recipe directory containing seed_* directories (default: MER FIPT (LaBraM) valence)')
    p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--selection-metric',choices=['balanced_accuracy'], default='balanced_accuracy',
                   help='Re-select epochs from logs; this does not replace the saved checkpoint')
    args=p.parse_args()
    rows,result=summarize(args.results,args.allow_partial,args.selection_metric)
    with (args.results/'summary.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.results/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    for row in rows:
        print(json.dumps(row, indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
