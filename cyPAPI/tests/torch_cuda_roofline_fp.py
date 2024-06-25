#!/usr/bin/env python3

# import necessary libraries
from cypapi import *

import torch

perrno = PAPI_Error["PAPI_OK"]

# size of tensors
m_rows = 1000
n_cols = 1000

# check to see if a device is available
if torch.cuda.is_available():
    unit = "cuda_7011"
else:
    raise Exception("NVIDIA device needed.")

try:
    # initialize cyPAPI
    cyPAPI_library_init()

    # check to see if cyPAPI was successfully initialized
    if cyPAPI_is_initialized() != 1:
        raise ValueError( "cyPAPI has not been initialized.\n" )

    # create  a cyPAPI EventSet 
    cuda_eventset = CyPAPI_EventSet()

    # add cuda native events to the EventSet
    cuda_ntv_evt1 = cyPAPI_event_name_to_code("cuda_7011:::smsp__sass_thread_inst_executed_op_fadd_pred_on.sum:device=0");
    cuda_eventset.add_event(cuda_ntv_evt1)
    cuda_ntv_evt2 = cyPAPI_event_name_to_code("cuda_7011:::smsp__sass_thread_inst_executed_op_fmul_pred_on.sum:device=0");
    cuda_eventset.add_event(cuda_ntv_evt2)
    cuda_ntv_evt3 = cyPAPI_event_name_to_code("cuda_7011:::smsp__sass_thread_inst_executed_op_ffma_pred_on.sum:device=0");
    cuda_eventset.add_event(cuda_ntv_evt3)
    cuda_ntv_evt4 = cyPAPI_event_name_to_code("cuda_7011:::smsp__sass_thread_inst_executed_op_dadd_pred_on.sum:device=0");
    cuda_eventset.add_event(cuda_ntv_evt4)
    cuda_ntv_evt5 = cyPAPI_event_name_to_code("cuda_7011:::smsp__sass_thread_inst_executed_op_dmul_pred_on.sum:device=0");
    cuda_eventset.add_event(cuda_ntv_evt5)
    cuda_ntv_evt6 = cyPAPI_event_name_to_code("cuda_7011:::smsp__sass_thread_inst_executed_op_dfma_pred_on.sum:device=0");
    cuda_eventset.add_event(cuda_ntv_evt6)
    cuda_ntv_evt7 = cyPAPI_event_name_to_code("cuda_7011:::smsp__sass_thread_inst_executed_op_hadd_pred_on.sum:device=0");
    cuda_eventset.add_event(cuda_ntv_evt7)
    cuda_ntv_evt8 = cyPAPI_event_name_to_code("cuda_7011:::smsp__sass_thread_inst_executed_op_hmul_pred_on.sum:device=0");
    cuda_eventset.add_event(cuda_ntv_evt8)
    cuda_ntv_evt9 = cyPAPI_event_name_to_code("cuda_7011:::smsp__sass_thread_inst_executed_op_hfma_pred_on.sum:device=0");
    cuda_eventset.add_event(cuda_ntv_evt9)
 
    # start counting hardware events in the created EventSet
    cuda_eventset.start()
    
    # create tensors for computation
    matrix_A = torch.rand( m_rows, n_cols, device = 'cuda' )
    matrix_B = torch.rand( m_rows, n_cols, device = 'cuda' )
    # perform matrix multiplication
    result_tensor = torch.mm( matrix_A, matrix_B )

    # transfer results to cpu
    result_cpu = result_tensor.detach().cpu()
    
    # stop counting hardware events in the created EventSet
    hw_counts = cuda_eventset.stop()
except:
    perrno = PAPI_Error["PAPI_EINVAL"]

# output if Cuda component has been successfully built
if perrno == PAPI_Error["PAPI_OK"]:
    # show number of available devices
    print( "Number of available devices: ", torch.cuda.device_count() )
    # show device name
    print( "Device Name: ", torch.cuda.get_device_name( 'cuda' ) )
    # counts for cuda native event 
    print( f"Hardware Counts for fadd:", hw_counts[0])
    print( f"Hardware Counts for fmul:", hw_counts[1])
    print( f"Hardware Counts for ffma:", hw_counts[2])
    print( f"Hardware Counts for dadd:", hw_counts[3])
    print( f"Hardware Counts for dmul:", hw_counts[4])
    print( f"Hardware Counts for dfma:", hw_counts[5])
    print( f"Hardware Counts for hadd:", hw_counts[6])
    print( f"Hardware Counts for hmul:", hw_counts[7])
    print( f"Hardware Counts for hfma:", hw_counts[8])
   # print( f"Hardware Counts for lts__t_sectors_op_write:", hw_counts[3])
    print("\033[0;32mPASSED\033[0m")
else:
    print("\033[0;31mFAILED\033[0m");
