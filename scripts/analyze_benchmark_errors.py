"""US-05-04: deterministic error inventory and figures from saved outputs only.

No model, training, calibration or inference module is imported. RGB/GT originals
and large NPZ remain external. Run --help for the inventory/fetch/render/check steps.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import urllib.parse
import urllib.request
from zipfile import ZipFile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/data/us-05-04'
REGISTRY_SHA = '41dd9d26e68840ca5d104062605d967e72a2d3907937e8ed6745743034c96b6a'
ARCHIVES = {
    'kaggle-001': ('US0503_ARCHIVE', 'benchmark-us0503/', '3f847e7bcbda2816ebb5ef4f41cd52a6720bd758b877a4a8f6f753d9ff7fcd45'),
    'kaggle-002': ('US0503_ARCHIVE_002', 'benchmark-us0503-remaining/', '85d9ff82c22126c181431dda845efeea56578ca66809d53e60b5ed61da4fe879'),
}
SIZE = (256, 256)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def read_json(path):
    return json.loads(path.read_bytes())


def csv_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode('utf-8'), newline='')))


def confined(root, relative):
    p = PurePosixPath(relative)
    if (not relative or p.is_absolute() or p.as_posix() != relative or '..' in p.parts
            or '\\' in relative or ':' in relative or any(ord(c) < 32 for c in relative)):
        raise ValueError(f'Unsafe path: {relative!r}')
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes root: {relative}')
    return result


def classify(label, score, threshold):
    if label not in (0, 1) or not math.isfinite(score) or not math.isfinite(threshold):
        raise ValueError('Invalid label, score or threshold')
    return ('TP' if label else 'FP') if score > threshold else ('FN' if label else 'TN')


def load_records(root=ROOT):
    """Bind pinned benchmark bytes to frozen thresholds and exact Test references."""
    spec = importlib.util.spec_from_file_location('us0503_summary', root / 'scripts/summarize_benchmark.py')
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    originals = previous.read_rows(root)  # All six export anchors + CSV/JSON hashes.
    registry_dir = root / 'experiments/calibration/six-thresholds'
    integrity = (registry_dir / 'integrity.json').read_bytes()
    if sha(integrity) != REGISTRY_SHA:
        raise ValueError('Frozen threshold registry anchor mismatch')
    threshold_data = (registry_dir / 'thresholds.json').read_bytes()
    if sha(threshold_data) != json.loads(integrity)['files']['thresholds.json']:
        raise ValueError('Frozen thresholds changed')
    frozen = {f"{r['model_family']}-{r['category']}": r for r in json.loads(threshold_data)['thresholds']}
    records = []
    for item in originals:
        instance, category = item['instance'], item['category']
        folder = (root / item['source_metrics']).parent
        report = read_json(folder / 'metrics.json')
        run = report['run']
        threshold = frozen[instance]['threshold']
        if threshold.hex() != run['threshold_hex'] or threshold.hex() != item['threshold_hex']:
            raise ValueError(f'Frozen threshold mismatch: {instance}')
        reference_bytes = (root / f'docs/data/us-02-03/{category}_manifest.csv').read_bytes()
        if sha(reference_bytes) != run['dataset_manifest_sha256']:
            raise ValueError(f'Dataset manifest mismatch: {instance}')
        refs = {r['image_path']: r for r in csv_rows(reference_bytes) if r['split'] == 'test'}
        predictions = csv_rows((folder / 'predictions.csv').read_bytes())
        manifest = read_json(folder / 'manifest.json')
        if len(predictions) != len(refs) or {r['image_path'] for r in predictions} != set(refs):
            raise ValueError(f'Incomplete/duplicate Test coverage: {instance}')
        for prediction, artifact in zip(predictions, manifest['artifacts'], strict=True):
            path = prediction['image_path']
            ref = refs[path]
            confined(root, path)  # Validate canonical identity, without reading RGB here.
            if (prediction['label'] != ref['label'] or artifact['image_path'] != path
                    or artifact['sha256'] != prediction['artifact_sha256']
                    or artifact['file'] != prediction['artifact_file']):
                raise ValueError(f'Prediction/reference mismatch: {instance}/{path}')
            label, score = int(prediction['label']), float(prediction['score'])
            records.append({
                'instance': instance, 'model_family': item['model_family'], 'category': category,
                'defect_type': ref['defect_type'], 'image_path': path, 'mask_path': ref['mask_path'],
                'label': label, 'score': score, 'threshold': threshold, 'threshold_hex': threshold.hex(),
                'classification': classify(label, score, threshold), 'signed_margin': score - threshold,
                'prediction_index': int(prediction['index']), 'artifact_file': artifact['file'],
                'artifact_sha256': artifact['sha256'], 'source_metrics': item['source_metrics'],
                'source_manifest_sha256': item['source_manifest_sha256'],
            })
        counts = Counter(r['classification'] for r in records if r['instance'] == instance)
        if any(counts[k.upper()] != item[k] for k in ('tp', 'tn', 'fp', 'fn')):
            raise ValueError(f'Confusion matrix mismatch: {instance}')
    return records


def select_cases(records):
    """One lower median of absolute margin per nonempty error/defect stratum.

    Margins are ranked only within the same instance; paths break ties. Selection
    does not inspect RGBs/maps. Both families are then paired on each chosen path.
    """
    groups = defaultdict(list)
    for row in records:
        if row['classification'] in ('FP', 'FN'):
            groups[(row['instance'], row['classification'], row['defect_type'])].append(row)
    chosen = defaultdict(list)
    for (instance, error, defect), rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda r: (abs(r['signed_margin']), r['image_path']))
        rank = (len(ordered) - 1) // 2
        row = ordered[rank]
        chosen[row['image_path']].append({'instance': instance, 'error': error, 'defect_type': defect,
                                         'stratum_count': len(rows), 'selected_rank_1based': rank + 1})
    cases = []
    for index, (path, reasons) in enumerate(sorted(chosen.items()), 1):
        paired = sorted((r for r in records if r['image_path'] == path), key=lambda r: r['model_family'])
        if len(paired) != 2 or {r['model_family'] for r in paired} != {'cae', 'patchcore'}:
            raise ValueError(f'Missing paired prediction: {path}')
        if len({(r['label'], r['mask_path']) for r in paired}) != 1:
            raise ValueError(f'Paired reference mismatch: {path}')
        cases.append({'case_id': f'case-{index:02d}', 'image_path': path, 'mask_path': paired[0]['mask_path'],
                      'selection_reasons': reasons, 'predictions': paired})
    return cases


def inventory_outputs(records):
    cases = select_cases(records)
    errors = [r for r in records if r['classification'] in ('FP', 'FN')]
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(errors)
    by_instance = []
    for instance in sorted({r['instance'] for r in records}):
        rows = [r for r in records if r['instance'] == instance]
        counts = Counter(r['classification'] for r in rows)
        strata = []
        for defect in sorted({r['defect_type'] for r in rows}):
            subset = [r for r in rows if r['defect_type'] == defect]
            c = Counter(r['classification'] for r in subset)
            strata.append({'defect_type': defect, 'n_images': len(subset), **{k: c[k] for k in ('TP','TN','FP','FN')}})
        by_instance.append({'instance': instance, 'n_images': len(rows),
                            **{k: counts[k] for k in ('TP','TN','FP','FN')}, 'by_defect_type': strata})
    return {
        'errors.csv': stream.getvalue().encode('utf-8'),
        'selection.json': json_bytes({'schema_version':'1.0',
            'policy':'lower_median_absolute_threshold_distance_per_instance_error_defect',
            'tie_break':'lexicographic image_path', 'rank':'floor((stratum_count - 1) / 2), zero-based',
            'selection_uses_images_or_maps':False, 'cases':cases}),
        'error-summary.json': json_bytes({'schema_version':'1.0','n_predictions':len(records),
            'n_errors':len(errors),'n_selected_paths':len(cases),'decision_rule':'score > threshold',
            'threshold_registry_integrity_sha256':REGISTRY_SHA,'instances':by_instance,
            'pixel_classification':'No calibrated pixel threshold; no pixel FP/FN, IoU or Dice inferred from image thresholds'}),
    }


def inventory(check=False):
    outputs = inventory_outputs(load_records())
    for name, data in outputs.items():
        path = EVIDENCE / name
        if check:
            if path.read_bytes() != data:
                raise ValueError(f'Inventory differs: {path}')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return json.loads(outputs['selection.json'])['cases']


def required_paths(cases):
    return {p for c in cases for p in (c['image_path'],c['mask_path']) if p}


def verify_source(data, metadata):
    if len(data) != metadata['size_bytes'] or sha(data) != metadata['sha256']:
        raise ValueError('Original source PNG bytes differ from pinned provenance')
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if (image.width,image.height) != (metadata['width'],metadata['height']):
            raise ValueError('Original source dimensions differ')
        decoded = image.convert(metadata['decoded_mode'])
        if sha(decoded.tobytes()) != metadata['decoded_sha256']:
            raise ValueError('Decoded source pixels differ')
        return decoded


def unwrap_download(data,relative,expected_size):
    """Kaggle may wrap one PNG in a ZIP; never extract arbitrary archive paths."""
    if data.startswith(b'PK'):
        with ZipFile(io.BytesIO(data)) as archive:
            entries=archive.infolist()
            if (len(entries)!=1 or entries[0].filename not in (relative,PurePosixPath(relative).name)
                    or entries[0].file_size!=expected_size):
                raise ValueError('Unexpected per-file ZIP wrapper')
            data=archive.read(entries[0])
    if not data.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('Expected original PNG; authentication or availability may have changed')
    return data


def fetch_images(dataset_root, cases):
    """Fetch only pinned selected files; never accept changed bytes on re-download."""
    source = read_json(EVIDENCE / 'source-images.json')
    if set(source['files']) != required_paths(cases):
        raise ValueError('Source inventory does not match selected paths')
    for relative, metadata in source['files'].items():
        path = confined(dataset_root, relative)
        if path.is_file():
            verify_source(path.read_bytes(), metadata)
            continue
        expected = ('https://www.kaggle.com/api/v1/datasets/download/ipythonx/mvtec-ad/'
                    + urllib.parse.quote(relative, safe='') + '?datasetVersionNumber=2')
        if metadata['url'] != expected:
            raise ValueError('Unexpected source URL')
        with urllib.request.urlopen(expected, timeout=30) as response:
            data = response.read(metadata['size_bytes'] + 65536)
        data=unwrap_download(data,relative,metadata['size_bytes'])
        verify_source(data, metadata)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f'Downloaded and verified {relative}', flush=True)


def aligned_source(dataset_root, case, source):
    rgb = verify_source(confined(dataset_root,case['image_path']).read_bytes(),source['files'][case['image_path']])
    if rgb.mode != 'RGB':
        raise ValueError('Expected RGB')
    if case['mask_path']:
        raw = verify_source(confined(dataset_root,case['mask_path']).read_bytes(),source['files'][case['mask_path']])
        if raw.size != rgb.size:
            raise ValueError('RGB/GT native dimensions differ')
        # Match src/data/mvtec_dataset.py + transforms.py without importing torch.
        mask = np.asarray(raw.convert('L').point(lambda p:255 if p>127 else 0)
                          .resize(SIZE,Image.Resampling.NEAREST)) > 127
    else:
        mask = np.zeros(SIZE,dtype=bool)
    return np.asarray(rgb.resize(SIZE,Image.Resampling.BILINEAR)),mask


def read_spatial(archives, row, expected_mask):
    batch = PurePosixPath(row['source_metrics']).parts[2]
    data = archives[batch].read(ARCHIVES[batch][1] + row['instance'] + '/' + row['artifact_file'])
    if sha(data) != row['artifact_sha256']:
        raise ValueError('NPZ hash differs')
    with np.load(io.BytesIO(data),allow_pickle=False) as arrays:
        if len(arrays.files)!=2 or set(arrays.files)!={'mask','anomaly_map'}:
            raise ValueError('Unexpected NPZ arrays')
        mask, amap = arrays['mask'],arrays['anomaly_map'].copy()
        if mask.dtype!=np.uint8 or mask.shape!=SIZE or not np.isin(mask,[0,1]).all():
            raise ValueError('Invalid archived mask')
        if amap.shape!=SIZE or amap.dtype!=np.float64 or not np.isfinite(amap).all():
            raise ValueError('Invalid archived anomaly map')
        if not np.array_equal(mask,expected_mask):
            raise ValueError('Raw source mask does not exactly match archived mask')
        if bool(mask.any()) != bool(row['label']):
            raise ValueError('Label/mask mismatch')
    return amap


def spatial_diagnostics(amap,mask):
    """Descriptive localization observations only, without a pixel decision threshold."""
    y,x = np.unravel_index(np.argmax(amap),amap.shape)
    result={'positive_pixels':int(mask.sum()),'positive_fraction':float(mask.mean()),
            'map_min':float(amap.min()),'map_max':float(amap.max()),
            'argmax_yx':[int(y),int(x)],'argmax_in_gt':bool(mask[y,x]) if mask.any() else None,
            'mean_inside_gt':float(amap[mask].mean()) if mask.any() else None,
            'mean_outside_gt':float(amap[~mask].mean()) if (~mask).any() else None}
    return result


def render_case(case, rgb, mask, maps, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colors
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':10,
                         'figure.facecolor':'white','savefig.facecolor':'white'})
    fig, axes = plt.subplots(2,4,figsize=(13.6,8.5),dpi=140)
    fig.subplots_adjust(left=.045,right=.95,bottom=.16,top=.83,wspace=.08,hspace=.44)
    path = case['image_path']
    fig.suptitle(f"{case['case_id']}  |  {path}",fontsize=17,fontweight='bold',y=.97)
    why='; '.join(f"{r['instance']} {r['error']} : rang {r['selected_rank_1based']}/{r['stratum_count']}" for r in case['selection_reasons'])
    fig.text(.5,.915,'Sélection : médiane inférieure de la distance au seuil dans chaque strate',ha='center',fontsize=10)
    fig.text(.5,.89,why,ha='center',fontsize=10)
    for j,row in enumerate(case['predictions']):
        amap=maps[row['model_family']]
        lo,hi=float(amap.min()),float(amap.max())
        if hi==lo: hi=lo+1.0
        norm=colors.Normalize(vmin=lo,vmax=hi)
        axes[j,0].imshow(rgb,interpolation='nearest')
        title=f"{row['model_family'].upper()} : {row['classification']} (label {row['label']})\nscore {row['score']:.6g} ; seuil {row['threshold']:.6g}"
        axes[j,0].set_title(title,loc='left',pad=8)
        im=axes[j,1].imshow(amap,cmap='magma',norm=norm,interpolation='nearest')
        axes[j,1].set_title('Carte archivée — valeurs brutes')
        bar=fig.colorbar(im,ax=axes[j,1],fraction=.046,pad=.025)
        bar.ax.tick_params(labelsize=8)
        axes[j,2].imshow(mask,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        axes[j,2].set_title('Masque réel (blanc = défaut)' if mask.any() else 'Masque réel vide (normale)')
        axes[j,3].imshow(rgb,interpolation='nearest')
        axes[j,3].imshow(amap,cmap='magma',norm=norm,alpha=.42,interpolation='nearest')
        if mask.any():
            axes[j,3].contour(mask.astype(float),levels=[.5],colors=['#00ffff'],linewidths=[1.1])
        axes[j,3].set_title('Superposition : carte + contour GT')
        for ax in axes[j]:ax.set_axis_off()
    fig.text(.05,.09,'RGB bilinéaire + masque nearest : 256 × 256, sans recadrage ni rotation. Contour cyan = vérité terrain.',fontsize=10)
    fig.text(.05,.061,'Échelle min–max propre à chaque carte, réservée à l’affichage ; les couleurs ne comparent pas les scores entre modèles.',fontsize=9)
    fig.text(.05,.035,'Aucun seuil pixel appliqué. Sources : MVTec AD / ipythonx v2 ; exports Kaggle US-05-03. Images dérivées : CC BY-NC-SA 4.0.',fontsize=9)
    target.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(target,dpi=140,metadata={'Software':'IndustrialDefect AI US-05-04'})
    plt.close(fig)


def render(dataset_root, output_dir, cases):
    source=read_json(EVIDENCE/'source-images.json')
    if set(source['files']) != required_paths(cases):
        raise ValueError('Unexpected source inventory')
    diagnostics=[]
    with ExitStack() as stack:
        archives={}
        for batch,(env,prefix,digest) in ARCHIVES.items():
            if not os.environ.get(env):
                raise ValueError(f'Missing external archive: {env}')
            path=Path(os.environ[env])
            if sha(path.read_bytes())!=digest:
                raise ValueError(f'Archive anchor mismatch: {batch}')
            archives[batch]=stack.enter_context(ZipFile(path))
        for case in cases:
            rgb,mask=aligned_source(dataset_root,case,source)
            maps={r['model_family']:read_spatial(archives,r,mask) for r in case['predictions']}
            filename=f"figures/{case['case_id']}.png"
            render_case(case,rgb,mask,maps,output_dir/filename)
            diagnostics.append({'case_id':case['case_id'],'image_path':case['image_path'],
                                'aligned_rgb_sha256':sha(rgb.tobytes()),'aligned_mask_sha256':sha(mask.astype(np.uint8).tobytes()),
                                'source_masks_equal_both_exports':True,
                                'figures':[filename],
                                'models':{family:spatial_diagnostics(amap,mask) for family,amap in maps.items()}})
    (output_dir/'spatial-diagnostics.json').write_bytes(json_bytes({'schema_version':'1.0','pixel_threshold':None,'cases':diagnostics}))
    import matplotlib
    import PIL
    outputs={'schema_version':'1.0','renderer':{'python':'.'.join(map(str,sys.version_info[:3])),
             'numpy':np.__version__,'Pillow':PIL.__version__,'matplotlib':matplotlib.__version__,
             'font':'DejaVu Sans bundled with matplotlib','backend':'Agg','dpi':140},
             'inputs':{name:sha((EVIDENCE/name).read_bytes()) for name in ('selection.json','source-images.json')},
             'files':{}}
    for relative in ['spatial-diagnostics.json']+[f"figures/{c['case_id']}.png" for c in cases]:
        data=(output_dir/relative).read_bytes()
        outputs['files'][relative]={'sha256':sha(data),'size_bytes':len(data)}
    (output_dir/'figures-manifest.json').write_bytes(json_bytes(outputs))


def verify_figures():
    manifest=read_json(EVIDENCE/'figures-manifest.json')
    for name,digest in manifest['inputs'].items():
        if sha((EVIDENCE/name).read_bytes())!=digest:
            raise ValueError('Figure input hash mismatch')
    for name,metadata in manifest['files'].items():
        data=confined(EVIDENCE,name).read_bytes()
        if sha(data)!=metadata['sha256'] or len(data)!=metadata['size_bytes']:
            raise ValueError(f'Figure artifact differs: {name}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('inventory','fetch','render','check','reproduce'))
    parser.add_argument('--dataset-root',type=Path,default=os.environ.get('US0504_DATASET'))
    args=parser.parse_args()
    cases=inventory(check=args.command!='inventory')
    if args.command=='fetch':
        if args.dataset_root is None: parser.error('--dataset-root or US0504_DATASET required')
        fetch_images(Path(args.dataset_root),cases)
    elif args.command in ('render','reproduce'):
        if args.dataset_root is None: parser.error('--dataset-root or US0504_DATASET required')
        if args.command=='render':render(Path(args.dataset_root),EVIDENCE,cases)
        else:
            verify_figures()
            with tempfile.TemporaryDirectory(prefix='us0504-reproduce-') as temporary:
                dest=Path(temporary)
                render(Path(args.dataset_root),dest,cases)
                for name in read_json(EVIDENCE/'figures-manifest.json')['files']:
                    if (dest/name).read_bytes()!=(EVIDENCE/name).read_bytes():
                        raise ValueError(f'Reproduction differs: {name}; use documented versions')
    elif args.command=='check':verify_figures()
    print(f'PASS: {args.command}; {len(cases)} selected paths; no inference or calibration')


if __name__=='__main__':main()
