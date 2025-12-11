import sys
import os
import numpy as np
import acl
import argparse

# Add current directory to sys.path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from preparedata import RecognizeSpeech_FromFile
from postprocess import SpeechPostProcess

# ACL Constants
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
NPY_FLOAT32 = 11

class AclResource:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.context = None
        self.stream = None
        self.run_mode = None

    def init(self):
        # Initialize ACL
        # Using ../src/acl.json as in the C++ sample
        acl_config_path = os.path.join(current_dir, "../src/acl.json")
        if not os.path.exists(acl_config_path):
            # Fallback or create empty if not exists, but it should exist
            print(f"[WARN] {acl_config_path} not found, passing empty string")
            acl_config_path = ""
            
        ret = acl.init(acl_config_path)
        if ret != 0:
            print(f"acl init failed, ret={ret}")
            return False
            
        ret = acl.rt.set_device(self.device_id)
        if ret != 0:
            print(f"set device failed, ret={ret}")
            return False
            
        self.context, ret = acl.rt.create_context(self.device_id)
        if ret != 0:
            print(f"create context failed, ret={ret}")
            return False
            
        self.stream, ret = acl.rt.create_stream()
        if ret != 0:
            print(f"create stream failed, ret={ret}")
            return False
            
        self.run_mode, ret = acl.rt.get_run_mode()
        if ret != 0:
            print(f"get run mode failed, ret={ret}")
            return False
            
        print("ACL init success")
        return True

    def release(self):
        if self.stream:
            acl.rt.destroy_stream(self.stream)
        if self.context:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()
        print("ACL release success")

class Model:
    def __init__(self, acl_resource, model_path):
        self.acl_resource = acl_resource
        self.model_path = model_path
        self.model_id = None
        self.model_desc = None
        self.input_dataset = None
        self.output_dataset = None
        self.input_buffers = []
        self.output_buffers = []

    def load(self):
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        if ret != 0:
            print(f"load model failed, ret={ret}")
            return False
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        if ret != 0:
            print(f"get model desc failed, ret={ret}")
            return False
        print("Model load success")
        return True

    def create_input(self, input_data):
        self.input_dataset = acl.mdl.create_dataset()
        
        input_size = input_data.nbytes
        input_ptr, ret = acl.rt.malloc(input_size, ACL_MEM_MALLOC_HUGE_FIRST)
        if ret != 0:
            print(f"malloc input failed, ret={ret}")
            return False
        
        # Copy data to device
        kind = ACL_MEMCPY_HOST_TO_DEVICE
        ptr_to_data = acl.util.numpy_to_ptr(input_data)
        ret = acl.rt.memcpy(input_ptr, input_size, ptr_to_data, input_size, kind)
        if ret != 0:
            print(f"memcpy input failed, ret={ret}")
            return False
            
        data_buffer = acl.create_data_buffer(input_ptr, input_size)
        _, ret = acl.mdl.add_dataset_buffer(self.input_dataset, data_buffer)
        if ret != 0:
            print(f"add input dataset buffer failed, ret={ret}")
            return False
        self.input_buffers.append(input_ptr)
        return True

    def create_output(self):
        self.output_dataset = acl.mdl.create_dataset()
        output_size = acl.mdl.get_num_outputs(self.model_desc)
        for i in range(output_size):
            buffer_size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            output_ptr, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            if ret != 0:
                print(f"malloc output failed, ret={ret}")
                return False
            data_buffer = acl.create_data_buffer(output_ptr, buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(self.output_dataset, data_buffer)
            if ret != 0:
                print(f"add output dataset buffer failed, ret={ret}")
                return False
            self.output_buffers.append(output_ptr)
        return True

    def execute(self):
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        if ret != 0:
            print(f"execute model failed, ret={ret}")
            return False
        return True

    def get_output(self):
        outputs = []
        num = acl.mdl.get_dataset_num_buffers(self.output_dataset)
        for i in range(num):
            data_buffer = acl.mdl.get_dataset_buffer(self.output_dataset, i)
            data_ptr = acl.get_data_buffer_addr(data_buffer)
            size = acl.get_data_buffer_size(data_buffer)
            
            output_host_ptr, ret = acl.rt.malloc_host(size)
            if ret != 0:
                print(f"malloc host failed, ret={ret}")
                return None
            
            ret = acl.rt.memcpy(output_host_ptr, size, data_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST)
            if ret != 0:
                print(f"memcpy output failed, ret={ret}")
                return None
            
            # Assuming float32 output based on postprocess.py
            data = acl.util.ptr_to_numpy(output_host_ptr, (size // 4,), NPY_FLOAT32)
            outputs.append(np.copy(data)) # Copy to ensure we own the data
            
            acl.rt.free_host(output_host_ptr)
            
        return outputs

    def release_dataset(self):
        if self.input_dataset:
            acl.mdl.destroy_dataset(self.input_dataset)
            self.input_dataset = None
        if self.output_dataset:
            acl.mdl.destroy_dataset(self.output_dataset)
            self.output_dataset = None
            
        for ptr in self.input_buffers:
            acl.rt.free(ptr)
        self.input_buffers = []
        
        for ptr in self.output_buffers:
            acl.rt.free(ptr)
        self.output_buffers = []

    def release(self):
        self.release_dataset()
        if self.model_id:
            acl.mdl.unload(self.model_id)
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)

def init_inference(model_path, device_id=0):
    """
    Initialize ACL resources and load the model.
    Returns (acl_resource, model) or (None, None) if failed.
    """
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None, None

    # Init ACL
    acl_resource = AclResource(device_id)
    if not acl_resource.init():
        return None, None

    # Load Model
    model = Model(acl_resource, model_path)
    if not model.load():
        acl_resource.release()
        return None, None
        
    return acl_resource, model

def release_inference(acl_resource, model):
    """
    Release ACL resources and model.
    """
    if model:
        model.release()
    if acl_resource:
        acl_resource.release()

def process_single_audio(model, audio_path):
    """
    Process a single audio file.
    Returns (text, pinyin) or (None, None) if failed.
    """
    print(f"Processing {audio_path}...")
    
    # Preprocess
    try:
        input_data, _ = RecognizeSpeech_FromFile(audio_path)
    except Exception as e:
        print(f"Preprocessing failed for {audio_path}: {e}")
        return None, None
    
    # Create Input/Output
    if not model.create_input(input_data):
        return None, None
    if not model.create_output():
        model.release_dataset()
        return None, None
        
    # Execute
    if not model.execute():
        model.release_dataset()
        return None, None
        
    # Get Output
    outputs = model.get_output()
    if outputs is None:
        model.release_dataset()
        return None, None
        
    # Postprocess
    result = outputs[0]
    
    txt = None
    pinyin = None
    try:
        txt, pinyin = SpeechPostProcess(result)
    except Exception as e:
        print(f"Postprocessing failed for {audio_path}: {e}")
    
    # Cleanup dataset for next iteration
    model.release_dataset()
    
    return txt, pinyin

def main(input_path, model_path):
    # Initialize
    acl_resource, model = init_inference(model_path)
    if not acl_resource or not model:
        print("Init inference failed")
        return

    # Determine files to process
    wav_files = []
    if os.path.isfile(input_path):
        if input_path.lower().endswith((".wav", ".mp3")):
            wav_files.append(input_path)
        else:
            print(f"Input file {input_path} is not a .wav or .mp3 file")
    elif os.path.isdir(input_path):
        if not os.path.exists(input_path):
             print(f"Data directory not found at {input_path}")
        else:
            for f in os.listdir(input_path):
                if f.lower().endswith((".wav", ".mp3")):
                    wav_files.append(os.path.join(input_path, f))
    else:
        print(f"Input path not found: {input_path}")
        release_inference(acl_resource, model)
        return

    if not wav_files:
        print(f"No .wav or .mp3 files found to process in {input_path}")
        release_inference(acl_resource, model)
        return
    
    # Process files
    import time
    # 计算如下代码耗时
    start_time = time.time()
    for wav_path in wav_files:
        txt, pinyin = process_single_audio(model, wav_path)
        if txt is not None:
            print(f"File: {os.path.basename(wav_path)}")
            print(f"Pinyin: {pinyin}")
            print(f"Text: {txt}")
            print("-" * 30)
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

    # Cleanup
    release_inference(acl_resource, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Speech to Text Inference')
    parser.add_argument('--input', '-i', default=os.path.join(current_dir, "../data"),
                        help='Path to input WAV file or directory containing WAV files')
    parser.add_argument('--model', '-m', default=os.path.join(current_dir, "../model/voice.om"),
                        help='Path to the OM model file')
    args = parser.parse_args()
    main(args.input, args.model)
