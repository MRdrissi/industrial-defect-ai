"""US-05-04 acceptance checks: no inference, no calibration and no network access."""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
import pytest

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('error_analysis',ROOT/'scripts/analyze_benchmark_errors.py')
analysis=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)
EVIDENCE=ROOT/'docs/data/us-05-04'


@pytest.fixture(scope='module')
def records():
    return analysis.load_records(ROOT)


def test_inventory_matches_all_saved_predictions_and_frozen_thresholds(records):
    assert len(records)==586
    outputs=analysis.inventory_outputs(records)
    for name,data in outputs.items():assert (EVIDENCE/name).read_bytes()==data
    summary=json.loads(outputs['error-summary.json'])
    assert summary['n_errors']==86 and summary['n_selected_paths']==13
    assert {r['instance']:(r['FP'],r['FN']) for r in summary['instances']}=={
        'cae-bottle':(0,26),'cae-hazelnut':(0,18),'cae-transistor':(0,34),
        'patchcore-bottle':(1,0),'patchcore-hazelnut':(0,0),'patchcore-transistor':(6,1)}
    for row in records:
        assert row['threshold'].hex()==row['threshold_hex']
        assert row['classification']==analysis.classify(row['label'],row['score'],row['threshold'])


def test_all_nonempty_error_strata_covered_without_invented_errors(records):
    cases=analysis.select_cases(records)
    selected={(r['instance'],r['error'],r['defect_type']) for c in cases for r in c['selection_reasons']}
    expected={(r['instance'],r['classification'],r['defect_type']) for r in records if r['classification'] in ('FP','FN')}
    assert selected==expected and len(selected)==13
    assert {c['image_path'].split('/')[0] for c in cases}=={'bottle','hazelnut','transistor'}
    assert not any(r['instance']=='patchcore-hazelnut' for c in cases for r in c['selection_reasons'])
    for case in cases:
        assert len(case['predictions'])==2
        assert {r['model_family'] for r in case['predictions']}=={'cae','patchcore'}
        for reason in case['selection_reasons']:
            chosen=next(r for r in case['predictions'] if r['instance']==reason['instance'])
            assert chosen['classification']==reason['error']
            stratum=[r for r in records if (r['instance'],r['classification'],r['defect_type'])==
                     (reason['instance'],reason['error'],reason['defect_type'])]
            ordered=sorted(stratum,key=lambda r:(abs(r['signed_margin']),r['image_path']))
            assert chosen['image_path']==ordered[(len(ordered)-1)//2]['image_path']


def test_selected_paths_are_stable_and_each_family_uses_same_reference(records):
    cases=analysis.select_cases(records)
    assert [c['image_path'] for c in cases]==[
        'bottle/test/broken_large/002.png','bottle/test/broken_small/015.png',
        'bottle/test/contamination/016.png','bottle/test/good/001.png',
        'hazelnut/test/crack/002.png','hazelnut/test/cut/001.png','hazelnut/test/hole/001.png',
        'transistor/test/bent_lead/008.png','transistor/test/cut_lead/005.png',
        'transistor/test/damaged_case/007.png','transistor/test/good/037.png',
        'transistor/test/misplaced/000.png','transistor/test/misplaced/002.png']
    for c in cases:
        assert {r['image_path'] for r in c['predictions']}=={c['image_path']}
        assert {r['mask_path'] for r in c['predictions']}=={c['mask_path']}
        assert len({r['label'] for r in c['predictions']})==1


@pytest.mark.parametrize('label,score,expected',[(0,1.0,'TN'),(1,1.0,'FN'),(0,1.00001,'FP'),(1,1.00001,'TP')])
def test_strict_threshold_rule_including_equality(label,score,expected):
    assert analysis.classify(label,score,1.0)==expected


@pytest.mark.parametrize('label,score,threshold',[(2,0.1,0.5),(1,float('nan'),0.5),(0,0.1,float('inf'))])
def test_invalid_classification_inputs_rejected(label,score,threshold):
    with pytest.raises(ValueError):analysis.classify(label,score,threshold)


def test_median_ties_are_deterministic_and_selection_does_not_use_spatial_data(records):
    # Change row order only; selection is determined by scores/thresholds and paths.
    assert analysis.select_cases(list(reversed(records)))==analysis.select_cases(records)
    sample=copy.deepcopy([r for r in records if r['image_path'].startswith('bottle/')])
    for r in sample:
        if r['instance']=='cae-bottle' and r['defect_type']=='broken_large':r['signed_margin']=-1.0
    choices=analysis.select_cases(sample)
    chosen=next(c for c in choices if any(r['instance']=='cae-bottle' and r['defect_type']=='broken_large' for r in c['selection_reasons']))
    paths=sorted(r['image_path'] for r in sample if r['instance']=='cae-bottle' and r['defect_type']=='broken_large' and r['classification']=='FN')
    assert chosen['image_path']==paths[(len(paths)-1)//2]


def test_missing_paired_prediction_rejected(records):
    selected=analysis.select_cases(records)[0]['image_path']
    incomplete=[r for r in records if not (r['image_path']==selected and r['model_family']=='patchcore')]
    with pytest.raises(ValueError,match='Missing paired'):analysis.select_cases(incomplete)


@pytest.mark.parametrize('relative',['../x.png','/x.png','C:/x.png','a\\b.png','a/./b.png','a//b.png'])
def test_path_traversal_and_noncanonical_paths_rejected(tmp_path,relative):
    with pytest.raises(ValueError):analysis.confined(tmp_path,relative)


def png_bytes():
    buffer=io.BytesIO()
    Image.new('RGB',(8,8),(10,20,30)).save(buffer,format='PNG')
    return buffer.getvalue()


def test_kaggle_raw_and_single_file_zip_preserve_original_bytes():
    data=png_bytes()
    assert analysis.unwrap_download(data,'bottle/test/good/000.png',len(data))==data
    buffer=io.BytesIO()
    with ZipFile(buffer,'w') as z:z.writestr('000.png',data)
    assert analysis.unwrap_download(buffer.getvalue(),'bottle/test/good/000.png',len(data))==data


@pytest.mark.parametrize('names',[['../000.png'],['unexpected.png'],['000.png','extra.txt']])
def test_unexpected_zip_members_rejected(names):
    data=png_bytes();buffer=io.BytesIO()
    with ZipFile(buffer,'w') as z:
        for name in names:z.writestr(name,data)
    with pytest.raises(ValueError):analysis.unwrap_download(buffer.getvalue(),'bottle/test/good/000.png',len(data))


def test_html_response_is_not_mistaken_for_source_image():
    with pytest.raises(ValueError,match='original PNG'):
        analysis.unwrap_download(b'<html>Sign in</html>','bottle/test/good/000.png',20)


def test_source_tampering_fails_before_render():
    data=png_bytes()
    meta={'size_bytes':len(data),'sha256':analysis.sha(data),'width':8,'height':8,
          'decoded_mode':'RGB','decoded_sha256':analysis.sha(Image.open(io.BytesIO(data)).tobytes())}
    assert analysis.verify_source(data,meta).size==(8,8)
    with pytest.raises(ValueError,match='bytes differ'):analysis.verify_source(data+b' ',meta)
    meta['width']=9
    with pytest.raises(ValueError,match='dimensions'):analysis.verify_source(data,meta)


def test_spatial_diagnostics_do_not_invent_pixel_classification():
    amap=np.arange(9,dtype=np.float64).reshape(3,3)
    empty=np.zeros((3,3),dtype=bool)
    normal=analysis.spatial_diagnostics(amap,empty)
    assert normal['argmax_in_gt'] is None and normal['mean_inside_gt'] is None
    mask=empty.copy();mask[-1,-1]=True
    abnormal=analysis.spatial_diagnostics(amap,mask)
    assert abnormal['argmax_in_gt'] is True and abnormal['argmax_yx']==[2,2]
    assert not {'pixel_fp','pixel_fn','iou','dice'} & set(abnormal)


def test_published_figures_provenance_and_dimensions():
    analysis.verify_figures()
    source=analysis.read_json(EVIDENCE/'source-images.json')
    cases=analysis.read_json(EVIDENCE/'selection.json')['cases']
    assert set(source['files'])==analysis.required_paths(cases)
    assert source['file_count']==24 and source['total_bytes']==13734107
    assert source['dataset']['version']==2 and source['raw_files_in_git'] is False
    figures=list((EVIDENCE/'figures').glob('*.png'))
    assert len(figures)==13
    for path in figures:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            assert image.size==(1904,1190)


@pytest.mark.skipif(not all(os.environ.get(k) for k in ('US0504_DATASET','US0503_ARCHIVE','US0503_ARCHIVE_002')),
                    reason='External RGB/GT and both benchmark ZIPs required; no download in tests')
def test_external_rgb_gt_and_both_saved_maps_align_exactly(records):
    source=analysis.read_json(EVIDENCE/'source-images.json')
    dataset=Path(os.environ['US0504_DATASET'])
    zips={b:ZipFile(os.environ[values[0]]) for b,values in analysis.ARCHIVES.items()}
    try:
        for case in analysis.select_cases(records):
            rgb,mask=analysis.aligned_source(dataset,case,source)
            assert rgb.shape==(256,256,3)
            for row in case['predictions']:
                amap=analysis.read_spatial(zips,row,mask)
                assert amap.shape==mask.shape
                wrong=mask.copy();wrong[0,0]=not wrong[0,0]
                with pytest.raises(ValueError,match='does not exactly match'):
                    analysis.read_spatial(zips,row,wrong)
    finally:
        for z in zips.values():z.close()
