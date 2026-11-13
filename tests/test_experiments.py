import json
import tempfile
import unittest
from pathlib import Path
from codes.multihop_qa.demo import fixture_documents,fixture_facts
from codes.multihop_qa.ingestion import chunk_documents,token_rechunk
from codes.multihop_qa.extraction import validate_fact
from codes.multihop_qa.retrieval import CandidateIndex
from codes.multihop_qa.mining import mine_paths
from codes.multihop_qa.baselines import baseline_plans,run_baseline
from codes.multihop_qa.annotations import create_pack,completed_reviews,source_review_pack
from codes.multihop_qa.quality import quality_report
from codes.multihop_qa.calibration import calibrate
from codes.multihop_qa.embeddings import embed_chunks
from codes.multihop_qa.sft import model_preflight,encode_supervised
from codes.multihop_qa.schema import uid,write_jsonl
from codes.multihop_qa.benchmark import answer_metrics
from codes.multihop_qa.document_parsers import page_interval
from codes.multihop_qa.providers import parse_model_object
from codes.multihop_qa.providers import JSONClient


class ExperimentTests(unittest.TestCase):
    def fixture(self):
        docs=fixture_documents()
        chunks=chunk_documents(docs)
        facts=[validate_fact(f,c) for f,c in zip(fixture_facts(chunks),chunks)]
        return docs,chunks,facts

    def test_real_baseline_single_and_adjacent(self):
        docs,chunks,_=self.fixture()
        self.assertEqual(len(baseline_plans(chunks,'single')),3)
        from codes.multihop_qa.ingestion import document_from_text
        c=chunk_documents([document_from_text('第一段说明方法。\n\n第二段说明作用。')])
        adjacent=baseline_plans(c,'adjacent')
        self.assertEqual(len(adjacent),1)
        self.assertEqual(adjacent[0]['evidence_chunk_ids'],[r['chunk_id'] for r in c])

    def test_baselines_do_not_cross_review_or_section(self):
        _,chunks,_=self.fixture()
        chunks[0]['needs_review']=True
        self.assertEqual(len(baseline_plans(chunks,'single')),2)
        self.assertFalse(baseline_plans(chunks,'adjacent'))

    def test_adjacent_baseline_does_not_jump_over_rejected_middle(self):
        from codes.multihop_qa.ingestion import document_from_text
        chunks=chunk_documents([document_from_text('甲段。\n\n乙段。\n\n丙段。')])
        chunks[1]['needs_review']=True
        self.assertFalse(baseline_plans(chunks,'adjacent'))

    def test_dense_baseline_needs_real_vectors(self):
        _,chunks,_=self.fixture()
        with self.assertRaises(ValueError):
            baseline_plans(chunks,'dense')

    def test_direct_multihop_control_selects_multiple_documents(self):
        _,chunks,_=self.fixture()
        plans=baseline_plans(chunks,'direct_multihop')
        mapping={c['chunk_id']:c for c in chunks}
        self.assertTrue(plans)
        self.assertTrue(all(len({mapping[c]['doc_id'] for c in p['evidence_chunk_ids']})==2 for p in plans))

    def test_implicit_candidates_survive_without_shared_entity(self):
        _,_,facts=self.fixture()
        for i,f in enumerate(facts):
            f['subject_id']='s'+str(i)
            f['object_id']='o'+str(i)
        index=CandidateIndex(facts,implicit=True)
        edges=[e for f in facts for e in index.recall(f)]
        paths=mine_paths(facts,edges)
        self.assertTrue(any(p['task_type']=='problem_solution' for p in paths))
        self.assertTrue(all(p['status']=='candidate_pending_semantic_verification' for p in paths))

    def test_embedding_cache_requires_matching_text_model(self):
        _,chunks,_=self.fixture()
        c=chunks[0]
        key=uid('embedding',['api','embedding-test','cls',c['text']])
        with tempfile.TemporaryDirectory() as root:
            cache=Path(root)/'cache.jsonl'
            write_jsonl(cache,[dict(cache_key=key,chunk_id='old',vector=[1.,0.],model='embedding-test')])
            report=embed_chunks([c],Path(root)/'out',model='embedding-test',backend='api',cache=cache)
            self.assertEqual(report['cache_hits'],1)
            self.assertEqual(report['generated'],0)
            vector=json.loads((Path(root)/'out'/'vectors.jsonl').read_text())
            self.assertEqual(vector['chunk_id'],c['chunk_id'])

    def pack(self,root):
        _,chunks,_=self.fixture()
        folder=Path(root)/'run'
        folder.mkdir()
        write_jsonl(folder/'chunks.jsonl',chunks)
        write_jsonl(folder/'qa.jsonl',[dict(qa_id='qa1',question='实际问题',answer='待核验答案',evidence_chunk_ids=[chunks[0]['chunk_id']])])
        pack=Path(root)/'pack'
        create_pack([dict(folder=str(folder),method='single')],pack)
        return pack

    def test_annotation_pack_never_fabricates_human_labels(self):
        with tempfile.TemporaryDirectory() as root:
            pack=self.pack(root)
            rows,total=completed_reviews(pack)
            self.assertEqual(rows,[])
            self.assertEqual(total,1)
            task=json.loads((pack/'tasks.jsonl').read_text(encoding='utf8'))
            self.assertNotIn('method',task)

    def test_no_labels_means_no_quality_claim(self):
        with tempfile.TemporaryDirectory() as root:
            pack=self.pack(root)
            report=quality_report(pack,Path(root)/'quality')
            self.assertEqual(report['status'],'awaiting_human_annotation')
            self.assertEqual(report['methods'],{})

    def test_source_review_pack_is_unlabelled(self):
        _,chunks,_=self.fixture()
        with tempfile.TemporaryDirectory() as root:
            report=source_review_pack(chunks,Path(root)/'pack')
            self.assertEqual(report['human_reviewed'],0)
            row=json.loads((Path(root)/'pack'/'source_reviews.jsonl').read_text(encoding='utf8').splitlines()[0])
            self.assertIsNone(row['decision'])

    def test_oversized_atomic_table_not_silently_truncated_for_embedding(self):
        _,chunks,_=self.fixture()
        chunks[0]['oversized']=True
        with tempfile.TemporaryDirectory() as root:
            report=embed_chunks([chunks[0]],Path(root)/'vectors',model='unused',backend='api')
            self.assertEqual(report['vectors'],0)
            self.assertEqual(len(report['skipped']),1)

    def calibration_rows(self):
        return [dict(record_id=str(i),score=score,label=label,split='val',human_reviewed=True,
                     reviewer='unit-test',source_group_ids=['validation'])
                for i,(score,label) in enumerate([(0.,0),(.1,0),(.6,1),(.9,1)])]

    def test_threshold_calibration_uses_validation(self):
        with tempfile.TemporaryDirectory() as root:
            rows=self.calibration_rows()
            report=calibrate(rows,Path(root)/'cal')
            self.assertTrue(report['calibrated'])
            self.assertEqual(report['validation']['f1'],1.)
            self.assertIsNone(report['test'])

    def test_calibration_rejects_test_only_or_source_leakage(self):
        with tempfile.TemporaryDirectory() as root:
            rows=self.calibration_rows()
            rows[0]['split']='test'
            with self.assertRaises(ValueError):
                calibrate(rows,Path(root)/'cal')
            for row in rows:
                row['split']='test'
            with self.assertRaises(ValueError):
                calibrate(rows,Path(root)/'cal2')

    def test_incomplete_download_is_not_trainable(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)
            (p/'config.json').write_text('{}')
            (p/'model.safetensors.incomplete').write_text('partial')
            report=model_preflight(p)
            self.assertFalse(report['ready'])
            self.assertIn('model.safetensors',report['missing'])

    def test_answer_only_training_masks_prompt(self):
        class Tokenizer:
            def apply_chat_template(self,messages,tokenize,add_generation_prompt):
                return ''.join(m['role']+':'+m['content']+';' for m in messages)+('assistant:' if add_generation_prompt else '')
            def encode(self,value,add_special_tokens=False):
                return list(value.encode())
        messages=[dict(role='user',content='Q'),dict(role='assistant',content='A')]
        record=encode_supervised(messages,Tokenizer(),100)
        self.assertTrue(record['labels'][0]==-100)
        self.assertEqual([v for v in record['labels'] if v!=-100],list(b'A;'))
        with self.assertRaises(ValueError):
            encode_supervised(messages,Tokenizer(),3)

    def test_token_budget_preserves_source_offsets(self):
        class Tokenizer:
            def encode(self,value,add_special_tokens=False):
                return list(value)
        docs,_,_=self.fixture()
        chunks=token_rechunk(docs,Tokenizer(),5)
        for c in chunks:
            self.assertLessEqual(c['token_count'],5)
            doc=next(d for d in docs if d['doc_id']==c['doc_id'])
            span=c['spans'][0]
            block=next(b for b in doc['blocks'] if b['block_id']==span['block_id'])
            self.assertEqual(c['text'],block['text'][span['start']:span['end']])

    def test_answer_metrics_are_explicit_surface_metrics(self):
        self.assertEqual(answer_metrics('答案。',['答案'])['exact_match'],1.)
        self.assertLess(answer_metrics('不同',['答案'])['character_f1'],1.)

    def test_page_ranges_are_inclusive(self):
        self.assertEqual(page_interval('2:4',10),(1,4))
        with self.assertRaises(ValueError):
            page_interval('0:4',10)

    def test_local_model_literal_normalization_never_executes_code(self):
        value,mode=parse_model_object("{'qa': None}")
        self.assertEqual(value,{'qa':None})
        self.assertEqual(mode,'safe_literal_normalization')
        with self.assertRaises(ValueError):
            parse_model_object("__import__('os').system('echo BAD')")

    def test_local_only_project_policy_blocks_external_document_calls(self):
        with self.assertRaisesRegex(ValueError,'local models only'):
            JSONClient('unused').ask('test','test',{'text':'must not leave this computer'})


if __name__=='__main__':
    unittest.main()
