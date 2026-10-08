import hashlib
import json
import pytest
from scripts.build_medical_comparison import checked_result,paired


def test_scorecard_rejects_changed_labels_and_duplicate_tasks(tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'examples':[{'case_id':'a','reference_answer':'B'}]}))
    result={'complete':True,'frozen':{'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'variants':['direct']},
        'records':[{'case_id':'a','variant':'direct','reference_answer':'B','prediction':None,'status':'failed','correct':False}]}
    path=tmp_path/'result.json';path.write_text(json.dumps(result))
    assert checked_result(path,manifest)['records'][0]['status']=='failed'
    result['records'].append(result['records'][0].copy());path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='Duplicated'):checked_result(path,manifest)
    result['records']=result['records'][:1];result['records'][0]['reference_answer']='A';path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='Changed answer'):checked_result(path,manifest)


def test_model_comparison_pairs_patient_ids_not_output_order():
    a={'records':[{'case_id':'a','variant':'direct','correct':True},{'case_id':'b','variant':'direct','correct':False}]}
    b={'records':[{'case_id':'b','variant':'direct','correct':True},{'case_id':'a','variant':'direct','correct':False}]}
    assert paired(a,b)=={'patients':2,'both_correct':0,'first_only_correct':1,'second_only_correct':1,'both_wrong':0}
    b['records'].pop()
    with pytest.raises(ValueError,match='Different patients'):paired(a,b)
