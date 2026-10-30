"""Meaningful smoke tests with synthetic RGB images and explicitly fake models.

These checks do not establish accuracy of real neural models or CZI decoding.
"""
import dataclasses
import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bal_pipeline as bp


class FakeModels:
    device_name="synthetic-test-no-neural-network"

    def __init__(self,neutrophil=False):
        self.segment_calls=0;self.expert_calls=0;self.classify_calls=0
        self.neutrophil=neutrophil

    def segment(self,image):
        self.segment_calls+=1
        return cv2.connectedComponents((image[:,:,1]<100).astype(np.uint8))[1]

    def predict(self,crops,expert=False):
        if expert:
            self.expert_calls+=1
            return np.tile(np.array([.1,.2,.7],np.float32),(len(crops),1))
        self.classify_calls+=1
        probs=np.full(8,.1/7,np.float32)
        probs[2 if self.neutrophil else 4]=.9
        return np.tile(probs,(len(crops),1))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.cfg=bp.Config("not-needed.pth","not-needed.pth","not-needed-cpsam",str(self.root/"results"),
                           tile_size=224,padding=8,min_area=20,batch_size=2,save_preview_pairs=1)

    def tearDown(self):
        self.temp.cleanup()

    def sample(self,name="test.tif",shape=(224,448),empty=False):
        image=np.full((*shape,3),210,np.uint8)
        if not empty:
            cv2.circle(image,(110,110),35,(120,50,150),-1)
        path=self.root/name
        tifffile.imwrite(path,image,photometric="rgb",metadata={"axes":"YXS"})
        return {"path":str(path),"filename":name,"animal_id":"001","treatment":"Test","timepoint":"6d"}

    def test_global_contours(self):
        mask=np.zeros((40,50),np.int32);mask[7:17,11:25]=1
        cell=bp.cells_from_masks(mask,100,200,min_area=1)[0]
        xy=np.concatenate(cell["contours_xy"])
        self.assertEqual(tuple(xy.min(axis=0)),(211,107))
        self.assertEqual(tuple(xy.max(axis=0)),(224,116))

    def test_large_contour_not_crop_clipped(self):
        mask=np.zeros((40,400),np.int32);mask[5:30,20:350]=1
        cell=bp.cells_from_masks(mask,0,0,min_area=1)[0]
        self.assertEqual(np.concatenate(cell["contours_xy"])[:,0].max(),349)

    def test_global_nms_and_far_cells(self):
        cells=[{"source_id":0,"area":100,"bbox":[0,2490,20,2510]},
               {"source_id":1,"area":90,"bbox":[0,2492,20,2512]},
               {"source_id":2,"area":80,"bbox":[0,7490,20,7510]}]
        kept=bp.deduplicate(cells)
        self.assertEqual([c["source_id"] for c in kept],[0,2])

    def test_nms_exact_threshold_and_empty(self):
        cells=[{"source_id":i,"area":9,"bbox":b} for i,b in enumerate([[0,0,3,3],[0,1,3,4]])]
        self.assertEqual(bp.bbox_iou(cells[0]["bbox"],cells[1]["bbox"]),.5)
        self.assertEqual(len(bp.deduplicate(cells)),2)
        self.assertEqual(bp.deduplicate([]),[])

    def test_axis_selection_and_rejection(self):
        selection,transpose=bp.axis_plan((1,1,1,10,20,3),"BSCYX0")
        image=np.zeros((1,1,1,10,20,3),np.uint8)
        self.assertEqual(image[selection].transpose(transpose).shape,(10,20,3))
        with self.assertRaises(ValueError):bp.axis_plan((2,10,20,3),"ZYXS")
        with self.assertRaises(ValueError):bp.axis_plan((10,20,2),"YXC")

    def test_edge_crop_shape_and_reflection(self):
        image=np.arange(30*40*3,dtype=np.int32).reshape(30,40,3).astype(np.uint8)
        result=bp.crop_at(image,0,0)
        self.assertEqual(result.shape,(224,224,3))
        np.testing.assert_array_equal(result[112,112],image[0,0])

    def test_particle_agglomerate_and_blank(self):
        image=np.full((224,224,3),240,np.uint8);image[50:110,50:110]=0
        self.assertEqual(bp.dark_structures(image,self.cfg)[0],1)
        self.assertEqual(bp.dark_structures(np.full_like(image,240),self.cfg)[0],0)

    def test_particle_reassignment_confidence_attribution(self):
        cell={"cy":112,"cx":112,"bbox":[80,80,140,140],"area":500,"border":False}
        probs=np.array([.01,.02,.01,.01,.02,.90,.01,.02])
        sample={"path":"test.czi","animal_id":"A","treatment":"X","timepoint":"1d"}
        with patch.object(bp,"dark_structures",return_value=(3,[])):
            record,_=bp.make_record(cell,probs,np.zeros((224,224,3),np.uint8),self.cfg,sample,1)
        self.assertEqual(record["Class"],"Macrophage_with_particles")
        self.assertEqual(record["Model_Predicted_Class"],"Monocyte")
        self.assertEqual(record["Model_Confidence"],.9)
        before=record.copy()
        self.assertEqual(bp.annotation_text(record),"Mac+p* | P:3")
        self.assertEqual(record,before)

    def test_compact_labels_preserve_all_classes_and_zero_particle_count(self):
        for category in bp.CLASSES:
            with self.subTest(category=category):
                row={"Model_Predicted_Class":category,"Class":category,
                     "Dark_Structure_Count":0,"Model_Confidence":.9,"Expert_Class":""}
                text=bp.annotation_text(row)
                self.assertTrue(text.startswith(bp.SHORT[category]))
                self.assertNotIn("%",text)
                if category=="Macrophage_with_particles":
                    self.assertEqual(text,"Mac+p | P:0")
                else:
                    self.assertNotIn("P:",text)
        row.update(Model_Predicted_Class="Macrophage_with_particles",
                   Class="Macrophage_with_particles",Dark_Structure_Count=12)
        self.assertEqual(bp.annotation_text(row),"Mac+p | P:12")

    def test_neutrophil_subtype_is_the_single_direct_label(self):
        row={"Model_Predicted_Class":"Neutrophil","Class":"Neutrophil",
             "Dark_Structure_Count":0,"Model_Confidence":.9,"Expert_Confidence":.7}
        expected=("Immature Neu","Mature Neu F+","Mature Neu F-")
        for expert,label in zip(bp.EXPERT_CLASSES,expected):
            with self.subTest(expert=expert):
                row["Expert_Class"]=expert
                self.assertEqual(bp.annotation_text(row),label)
        for unavailable in ("",None,np.nan,"unknown"):
            row["Expert_Class"]=unavailable
            self.assertEqual(bp.annotation_text(row),"Neu")
        del row["Expert_Class"]
        self.assertEqual(bp.annotation_text(row),"Neu")
        row.update(Class="Lymphocyte",Model_Predicted_Class="Lymphocyte",
                   Expert_Class="Immature_Neu")
        self.assertEqual(bp.annotation_text(row),"Lym")

    def test_rendered_label_is_single_and_matches_contour_colour(self):
        cell={"contours_xy":[[[85,80],[140,80],[140,140],[85,140]]]}
        for category in bp.CLASSES:
            expert_options=bp.EXPERT_CLASSES if category=="Neutrophil" else ("",)
            for expert in expert_options:
                with self.subTest(category=category,expert=expert):
                    canvas=np.full((224,224,3),210,np.uint8)
                    row={"Class":category,"Model_Predicted_Class":category,
                         "Model_Confidence":.9,"Dark_Structure_Count":0,
                         "BBox_MinX":85,"BBox_MinY":80,
                         "Expert_Class":expert,"Expert_Confidence":.7}
                    with patch.object(cv2,"putText",wraps=cv2.putText) as text_call,\
                         patch.object(cv2,"drawContours",wraps=cv2.drawContours) as contour_call:
                        bp.draw_annotation(canvas,cell,row,offset=(5,10))
                    text_call.assert_called_once()
                    contour_call.assert_called_once()
                    self.assertNotIn("%",text_call.call_args.args[1])
                    self.assertEqual(text_call.call_args.args[5],bp.COLORS[category])
                    self.assertEqual(text_call.call_args.args[5],contour_call.call_args.args[3])
                    # Antialiasing blends class colours with the white label background.
                    label_pixels=canvas[58:68,80:200]
                    self.assertTrue(np.any(label_pixels[:,:,0]!=label_pixels[:,:,1]))

    def test_full_pipeline_tiff_and_expert(self):
        sample=self.sample();models=FakeModels(neutrophil=True)
        result=bp.Pipeline(self.cfg,models).run([sample],mode="full")
        folder=Path(result["runs"][0]["folder"])
        cells=pd.read_csv(folder/"cells.csv")
        self.assertEqual(len(cells),1)
        self.assertEqual(cells.Expert_Class.iloc[0],bp.EXPERT_CLASSES[2])
        self.assertEqual(models.expert_calls,1)
        with tifffile.TiffFile(folder/"annotated_contours.tif") as reader:
            self.assertTrue(reader.is_bigtiff)
            self.assertTrue(reader.pages[0].is_tiled)
            self.assertEqual(reader.asarray().shape,(224,448,3))
            pixels=reader.asarray()
            self.assertTrue(np.any(pixels[:,:224] != tifffile.imread(sample["path"])[:,:224]))
            np.testing.assert_array_equal(pixels[:,300:],tifffile.imread(sample["path"])[:,300:])
        self.assertTrue((Path(result["batch_folder"])/"TIFF_OUTPUTS.csv").exists())
        self.assertTrue((Path(result["batch_folder"])/"ANIMAL_STATISTICS.csv").is_file())
        animals=pd.read_csv(Path(result["batch_folder"])/"ANIMAL_STATISTICS.csv",dtype={"Animal_ID":str})
        self.assertEqual(animals.Animal_ID.iloc[0],"001")

    def test_no_neutrophils_expert_not_called(self):
        sample=self.sample();models=FakeModels()
        result=bp.Pipeline(self.cfg,models).process(sample,mode="full")
        folder=Path(result["folder"])
        self.assertEqual(models.expert_calls,0)
        self.assertTrue(pd.read_csv(folder/"neutrophil_cells.csv").empty)
        report=json.loads((folder/"RUN_REPORT.json").read_text())
        self.assertEqual(report["expert_status"],"no_neutrophils")
        summary=pd.read_csv(folder/"slide_summary.csv").iloc[0]
        self.assertTrue(np.isnan(summary["Subtype_Immature_Neu_WithinNeutrophils_pct"]))

    def test_tiff_pixel_spacing_preserved(self):
        sample=self.sample()
        image=tifffile.imread(sample["path"])
        tifffile.imwrite(sample["path"],image,photometric="rgb",resolution=(10000,8000),resolutionunit="CENTIMETER")
        result=bp.Pipeline(self.cfg,FakeModels()).process(sample,mode="full")
        info=bp.inspect_slide(Path(result["folder"])/"annotated_contours.tif")
        self.assertAlmostEqual(info["pixel_size_um"]["X"],1)
        self.assertAlmostEqual(info["pixel_size_um"]["Y"],1.25)

    def test_empty_partial_visits_exactly_one_blank_tile(self):
        sample=self.sample(empty=True);models=FakeModels()
        result=bp.Pipeline(self.cfg,models).run([sample],mode="preview",preview_tiles=1)
        folder=Path(result["runs"][0]["folder"])
        cells=pd.read_csv(folder/"cells.csv")
        self.assertTrue(cells.empty)
        self.assertEqual(list(cells.columns),bp.CELL_COLUMNS)
        report=json.loads((folder/"RUN_REPORT.json").read_text())
        self.assertEqual(report["tiles_visited"],1)
        self.assertEqual(report["scope"],"partial")
        self.assertEqual(models.segment_calls,1)
        summary=pd.read_csv(folder/"slide_summary.csv").iloc[0]
        self.assertTrue(np.isnan(summary.WBC_Share_of_AllCells_pct))
        self.assertFalse((Path(result["batch_folder"])/"ANIMAL_STATISTICS.csv").exists())

    def test_preview_never_promoted_to_full(self):
        result=bp.Pipeline(self.cfg,FakeModels()).process(self.sample(shape=(224,224)),mode="preview",preview_tiles=10)
        report=json.loads((Path(result["folder"])/"RUN_REPORT.json").read_text())
        self.assertEqual(report["scope"],"partial")

    def test_resume_and_missing_tiff(self):
        sample=self.sample();models=FakeModels();pipe=bp.Pipeline(self.cfg,models)
        first=pipe.process(sample,mode="full");calls=models.segment_calls
        second=pipe.process(sample,mode="full")
        self.assertEqual(first["folder"],second["folder"])
        self.assertEqual(calls,models.segment_calls)
        (Path(first["folder"])/"annotated_contours.tif").unlink()
        third=pipe.process(sample,mode="full")
        self.assertNotEqual(first["folder"],third["folder"])
        self.assertGreater(models.segment_calls,calls)

    def test_modified_csv_not_reused(self):
        sample=self.sample();pipe=bp.Pipeline(self.cfg,FakeModels())
        result=pipe.process(sample,mode="full");folder=Path(result["folder"])
        report=json.loads((folder/"RUN_REPORT.json").read_text())
        self.assertTrue(bp.completed_run(folder,report["fingerprint"]))
        (folder/"cells.csv").write_text("truncated")
        self.assertFalse(bp.completed_run(folder,report["fingerprint"]))

    def test_failure_status_not_complete(self):
        class Broken(FakeModels):
            def segment(self,image):raise RuntimeError("Injected failure")
        sample=self.sample()
        with self.assertRaises(RuntimeError):bp.Pipeline(self.cfg,Broken()).process(sample,mode="full")
        manifests=list((self.root/"results").rglob("RUN_REPORT.json"))
        self.assertEqual(len(manifests),1)
        self.assertEqual(json.loads(manifests[0].read_text())["status"],"failed")

    def make_summary(self,animal,n,neu):
        sample={"path":f"{animal}_{n}.tif","animal_id":animal,"treatment":"T","timepoint":"6d"}
        rows=[]
        for i in range(n):
            record={c:np.nan for c in bp.CELL_COLUMNS}
            record.update(Class="Neutrophil" if i<neu else "Macrophage",Expert_Class=bp.EXPERT_CLASSES[2] if i<neu else "",
                          Dark_Structure_Count=0,Particle_Rule_Applied=False,Touches_Tile_Border=False,Model_Confidence=.9,
                          Expert_Confidence=.8 if i<neu else np.nan)
            rows.append(record)
        row=bp.summarise_cells(pd.DataFrame(rows,columns=bp.CELL_COLUMNS),sample,True)
        row["Scope"]="full"
        return row

    def test_animal_pooling_and_sample_sd(self):
        slides=pd.DataFrame([self.make_summary("A",10,9),self.make_summary("A",90,0),self.make_summary("B",10,10)])
        animals,groups=bp.aggregate_animals(slides)
        self.assertAlmostEqual(animals.iloc[0].BAL_Differential_Neutrophil_pct,9)
        row=groups[groups.Metric=="BAL_Differential_Neutrophil_pct"].iloc[0]
        self.assertEqual(row.N_Animals,2)
        self.assertAlmostEqual(row.Mean,54.5)
        self.assertAlmostEqual(row.SD,64.34671708797582)

    def test_aggregate_rejects_partial_and_conflicts(self):
        slides=pd.DataFrame([self.make_summary("A",10,9),self.make_summary("A",90,0)])
        slides.loc[1,"Treatment"]="other"
        with self.assertRaises(ValueError):bp.aggregate_animals(slides)
        slides.loc[1,"Treatment"]="T";slides.loc[1,"Scope"]="partial"
        with self.assertRaises(ValueError):bp.aggregate_animals(slides)

    def test_metadata_must_be_explicit_and_unique(self):
        sample=self.sample()
        with self.assertRaises(ValueError):bp.load_samples([sample["path"],sample["path"]])
        rows=bp.load_samples([sample["path"]])
        self.assertEqual(rows[0]["animal_id"],"")

    def test_missing_metadata_file_rejected(self):
        sample=self.sample()
        with self.assertRaises(FileNotFoundError):
            bp.load_samples([sample["path"]],self.root/"missing.csv")

    def test_uint16_not_silently_rescaled(self):
        path=self.root/"16bit.tif"
        tifffile.imwrite(path,np.zeros((20,30,3),np.uint16),photometric="rgb")
        with self.assertRaises(ValueError):bp.inspect_slide(path)

    def test_nan_inference_fails(self):
        class BadProb(FakeModels):
            def predict(self,crops,expert=False):
                return np.full((len(crops),8),np.nan)
        with self.assertRaises(RuntimeError):
            bp.Pipeline(self.cfg,BadProb()).process(self.sample(),mode="full")

    def test_empty_run_rejected(self):
        with self.assertRaises(ValueError):bp.Pipeline(self.cfg,FakeModels()).run([])

    def test_metadata_leading_zero_ids_not_merged(self):
        first=self.sample("one.tif");second=self.sample("two.tif")
        meta=self.root/"metadata.csv"
        pd.DataFrame([{"filename":"one.tif","animal_id":"001","treatment":"T","timepoint":"1d"},
                      {"filename":"two.tif","animal_id":"1","treatment":"T","timepoint":"1d"}]).to_csv(meta,index=False)
        result=bp.load_samples([first["path"],second["path"]],meta)
        self.assertEqual([s["animal_id"] for s in result],["001","1"])

    def test_cellpose_builtin_sentinel_not_a_relative_path(self):
        config=dataclasses.replace(self.cfg,cellpose_checkpoint="cpsam")
        self.assertEqual(config.cellpose_checkpoint,"cpsam")
        local=dataclasses.replace(self.cfg,cellpose_checkpoint="./models/cpsam")
        self.assertTrue(Path(local.cellpose_checkpoint).is_absolute())
        self.assertNotEqual(local.cellpose_checkpoint,"cpsam")

    def test_cellpose_library_cache_resolution_and_hash(self):
        weights=self.root/"cached_cpsam";weights.write_bytes(b"synthetic cached weights")
        fake=SimpleNamespace(models=SimpleNamespace(cache_CPSAM_model_path=lambda: str(weights)))
        classifier=self.root/"classifier.pth";classifier.write_bytes(b"classifier")
        expert=self.root/"expert.pth";expert.write_bytes(b"expert")
        config=dataclasses.replace(self.cfg,classifier_checkpoint=str(classifier),expert_checkpoint=str(expert),cellpose_checkpoint="cpsam")
        with patch.dict(sys.modules,{"cellpose":fake}),patch.object(bp.imd,"version",return_value="4.0.6"),\
             patch.object(bp,"execution_plan",return_value={"cellpose_device":"cpu","classifier_device":"cpu","cellpose_dtype":"float32"}):
            pipe=bp.Pipeline(config)
            self.assertEqual(pipe.resolved_cellpose_path,weights.resolve())
            self.assertEqual(pipe.model_hashes["cellpose"],bp.sha256(weights))
            old_hash=pipe.model_hashes["cellpose"]
            weights.write_bytes(b"changed cached weights")
            self.assertNotEqual(bp.Pipeline(config).model_hashes["cellpose"],old_hash)

    def test_cellpose_local_override_never_downloads(self):
        weights=self.root/"local_cpsam";weights.write_bytes(b"local")
        with patch.object(bp.imd,"version",side_effect=AssertionError("Unexpected library lookup")):
            self.assertEqual(bp.resolve_cellpose_checkpoint(str(weights)),weights.resolve())
            with self.assertRaises(FileNotFoundError):
                bp.resolve_cellpose_checkpoint(str(self.root/"missing_cpsam"))

    def test_cellpose_download_failure_is_actionable(self):
        def fail():raise OSError("network unavailable")
        fake=SimpleNamespace(models=SimpleNamespace(cache_CPSAM_model_path=fail))
        with patch.dict(sys.modules,{"cellpose":fake}),patch.object(bp.imd,"version",return_value="4.0.6"):
            with self.assertRaisesRegex(RuntimeError,"first download needs internet"):
                bp.resolve_cellpose_checkpoint("cpsam")

    def test_injected_models_never_download_cellpose(self):
        config=dataclasses.replace(self.cfg,cellpose_checkpoint="cpsam")
        with patch.object(bp,"resolve_cellpose_checkpoint",side_effect=AssertionError("Unexpected download")):
            bp.Pipeline(config,FakeModels())

    def test_cellpose_preflight_never_downloads(self):
        classifier=self.root/"classifier.pth";classifier.write_bytes(b"classifier")
        expert=self.root/"expert.pth";expert.write_bytes(b"expert")
        config=dataclasses.replace(self.cfg,classifier_checkpoint=str(classifier),expert_checkpoint=str(expert),cellpose_checkpoint="cpsam")
        info={"height":224,"width":224,"raw_bytes":224*224*3}
        fake_torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
        with patch.object(bp.importlib.util,"find_spec",return_value=object()),patch.dict(sys.modules,{"torch":fake_torch}),\
             patch.object(bp,"inspect_slide",return_value=info),patch.object(bp,"resolve_cellpose_checkpoint") as resolver:
            bp.preflight(config,[self.sample()])
            resolver.assert_not_called()

    def test_erythrocyte_retained_in_table_and_tiff(self):
        class EryModels(FakeModels):
            def predict(self,crops,expert=False):
                self.assert_not_expert=not expert
                if expert:raise AssertionError("Erythrocyte was sent to neutrophil expert")
                probs=np.full(8,.1/7,np.float32);probs[3]=.9
                return np.tile(probs,(len(crops),1))
        sample=self.sample();models=EryModels()
        result=bp.Pipeline(self.cfg,models).process(sample,mode="full")
        folder=Path(result["folder"])
        cells=pd.read_csv(folder/"cells.csv");row=cells.iloc[0]
        self.assertEqual(row.Class,"Erythrocyte")
        self.assertEqual(bp.annotation_text(row),"Ery")
        summary=pd.read_csv(folder/"slide_summary.csv").iloc[0]
        self.assertEqual(summary.Count_Erythrocyte,1)
        self.assertEqual(summary.Total_Detected_Cells,1)
        self.assertEqual(summary.Total_WBC,0)
        self.assertEqual(summary.Share_Erythrocyte_AllCells_pct,100)
        pixels=tifffile.imread(folder/"annotated_contours.tif")
        self.assertTrue(np.any(pixels!=tifffile.imread(sample["path"])))

    def test_auto_backend_change_does_not_reuse_previous_device_output(self):
        sample=self.sample()
        first_models=FakeModels();second_models=FakeModels()
        first_models.execution_plan={"cellpose_device":"cpu","classifier_device":"cpu","cellpose_dtype":"float32"}
        second_models.execution_plan={"cellpose_device":"mps","classifier_device":"mps","cellpose_dtype":"float32"}
        first=bp.Pipeline(self.cfg,first_models).process(sample,mode="full")
        second=bp.Pipeline(self.cfg,second_models).process(sample,mode="full")
        self.assertNotEqual(first["folder"],second["folder"])
        one=json.loads((Path(first["folder"])/"RUN_REPORT.json").read_text())
        two=json.loads((Path(second["folder"])/"RUN_REPORT.json").read_text())
        self.assertNotEqual(one["fingerprint"],two["fingerprint"])
        self.assertEqual(two["execution_plan"]["cellpose_device"],"mps")


class DeviceTests(unittest.TestCase):
    @staticmethod
    def torch_stub(cuda=False,mps=False,with_mps=True):
        return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:cuda),
            backends=SimpleNamespace(**({"mps":SimpleNamespace(is_available=lambda:mps)} if with_mps else {})))

    def test_automatic_device_priority(self):
        for cuda,mps,expected in [(True,True,"cuda"),(False,True,"mps"),(False,False,"cpu")]:
            with self.subTest(cuda=cuda,mps=mps):
                self.assertEqual(bp.select_device("auto",self.torch_stub(cuda,mps)),expected)

    def test_missing_mps_backend_and_explicit_cpu(self):
        self.assertEqual(bp.select_device("auto",self.torch_stub(with_mps=False)),"cpu")
        self.assertEqual(bp.select_device("cpu",self.torch_stub(True,True)),"cpu")

    def test_explicit_unavailable_accelerator_fails(self):
        for device in ("cuda","mps"):
            with self.assertRaises(RuntimeError):bp.select_device(device,self.torch_stub())

    def test_mps_float32_and_mixed_device_plan(self):
        config=bp.Config("a","b","cpsam","results",device="mps",classifier_device="cpu",cellpose_bfloat16=True)
        plan=bp.execution_plan(config,self.torch_stub(mps=True))
        self.assertEqual(plan["cellpose_device"],"mps")
        self.assertEqual(plan["classifier_device"],"cpu")
        self.assertEqual(plan["cellpose_dtype"],"float32")

    def test_cuda_retains_requested_bfloat16(self):
        config=bp.Config("a","b","cpsam","results",device="cuda",cellpose_bfloat16=True)
        self.assertEqual(bp.execution_plan(config,self.torch_stub(cuda=True))["cellpose_dtype"],"bfloat16")

    def test_native_models_mps_init_and_cpu_classifier_routing(self):
        events={"cellpose":None,"heads":[],"tensor_targets":[]}
        class Device(str):
            @property
            def type(self):return str(self)
        class Tensor:
            def __init__(self,array):self.array=np.asarray(array)
            def to(self,device):events["tensor_targets"].append(str(device));return self
            def cpu(self):return self
            def numpy(self):return self.array
            def __add__(self,other):return Tensor(self.array+(other.array if isinstance(other,Tensor) else other))
            __radd__=__add__
            def __truediv__(self,other):return Tensor(self.array/other)
        class Head:
            def __init__(self,n):self.n=n;self.target=None;self.strict=None;self.fp32=False
            def load_state_dict(self,state,strict):self.strict=strict
            def float(self):self.fp32=True;return self
            def to(self,device):self.target=str(device);return self
            def eval(self):return self
            def __call__(self,batch):return Tensor(np.zeros((len(batch.array),self.n),np.float32))
        def create_model(name,**kwargs):
            head=Head(kwargs["num_classes"]);events["heads"].append(head);return head
        def cellpose(**kwargs):
            events["cellpose"]=kwargs
            return SimpleNamespace(eval=lambda image,**options:(np.zeros(image.shape[:2],np.int32),None,None))
        torch=self.torch_stub(mps=True)
        torch.device=Device
        torch.load=lambda *a,**k:{"head.weight":"synthetic"}
        torch.from_numpy=Tensor
        torch.inference_mode=contextlib.nullcontext
        torch.rot90=lambda batch,k,axes:Tensor(np.rot90(batch.array,k,axes))
        def softmax(tensor,dim):
            values=np.exp(tensor.array);return Tensor(values/values.sum(axis=dim,keepdims=True))
        torch.softmax=softmax
        modules={"torch":torch,"timm":SimpleNamespace(create_model=create_model),
                 "cellpose.models":SimpleNamespace(CellposeModel=cellpose)}
        config=bp.Config("a","b","cpsam","results",device="mps",classifier_device="cpu",cellpose_bfloat16=True)
        with patch.dict(sys.modules,modules):
            models=bp.Models(config,resolved_cellpose_path=Path("synthetic-cpsam"))
            self.assertEqual(events["cellpose"]["device"].type,"mps")
            self.assertTrue(events["cellpose"]["gpu"])
            self.assertFalse(events["cellpose"]["use_bfloat16"])
            self.assertTrue(all(h.strict and h.fp32 and h.target=="cpu" for h in events["heads"]))
            crops=[np.full((224,224,3),128,np.uint8)]
            self.assertEqual(models.predict(crops).shape,(1,8))
            self.assertEqual(models.predict(crops,expert=True).shape,(1,3))
            self.assertEqual(events["tensor_targets"],["cpu","cpu"])
            self.assertEqual(models.segment(crops[0]).shape,(224,224))
            auto=dataclasses.replace(config,device="auto",classifier_device="same")
            accelerated=bp.Models(auto,resolved_cellpose_path=Path("synthetic-cpsam"))
            self.assertTrue(all(h.target=="mps" for h in events["heads"][-2:]))
            accelerated.predict(crops)
            self.assertEqual(events["tensor_targets"][-1],"mps")


if __name__=="__main__":
    unittest.main(verbosity=2)
