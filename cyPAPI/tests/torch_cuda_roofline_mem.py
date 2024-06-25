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
    #cuda_ntv_evt1 = cyPAPI_event_name_to_code("cuda_7011:::dram__sectors_read.sum:device=0");
    #cuda_eventset.add_event(cuda_ntv_evt1)
    #cuda_ntv_evt2 = cyPAPI_event_name_to_code("cuda_7011:::dram__sectors_write.sum:device=0");
    #cuda_eventset.add_event(cuda_ntv_evt2)
    #cuda_ntv_evt3 = cyPAPI_event_name_to_code("cuda_7011:::lts__t_sectors_op_read.sum:device=0");
    #cuda_eventset.add_event(cuda_ntv_evt3)
    #cuda_ntv_evt4 = cyPAPI_event_name_to_code("cuda_7011:::lts__t_sectors_op_write.sum:device=0");
    #cuda_eventset.add_event(cuda_ntv_evt4)
    #cuda_ntv_evt5 = cyPAPI_event_name_to_code("cuda_7011:::lts__t_sectors_op_atom.sum:device=0");
    #cuda_eventset.add_event(cuda_ntv_evt5)
    #cuda_ntv_evt6 = cyPAPI_event_name_to_code("cuda_7011:::lts__t_sectors_op_red.sum:device=0");
    #cuda_eventset.add_event(cuda_ntv_evt6)
   # cuda_ntv_evt7 = cyPAPI_event_name_to_code("cuda_7011:::l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum:device=0");
   # cuda_eventset.add_event(cuda_ntv_evt7)
   # cuda_ntv_evt8 = cyPAPI_event_name_to_code("cuda_7011:::l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum:device=0");
   # cuda_eventset.add_event(cuda_ntv_evt8)
   # cuda_ntv_evt9 = cyPAPI_event_name_to_code("cuda_7011:::l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum:device=0");
   # cuda_eventset.add_event(cuda_ntv_evt9)
   # cuda_ntv_evt10 = cyPAPI_event_name_to_code("cuda_7011:::l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum:device=0");
   # cuda_eventset.add_event(cuda_ntv_evt10)
   # cuda_ntv_evt11 = cyPAPI_event_name_to_code("cuda_7011:::l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum:device=0");
   # cuda_eventset.add_event(cuda_ntv_evt11)
   # cuda_ntv_evt12 = cyPAPI_event_name_to_code("cuda_7011:::l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum:device=0");
   # cuda_eventset.add_event(cuda_ntv_evt12)
    cuda_ntv_evt13 = cyPAPI_event_name_to_code("cuda_7011:::l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum:device=0");
    cuda_eventset.add_event(cuda_ntv_evt13)
    cuda_ntv_evt14 = cyPAPI_event_name_to_code("cuda_7011:::l1tex__t_set_accesses_pipe_lsu_mem_global_op_red.sum:device=0");
    cuda_eventset.add_event(cuda_ntv_evt14)
 
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
    print( f"Hardware Counts for dram__sectors_read:", hw_counts[0])
    print( f"Hardware Counts for dram__sectors_write:", hw_counts[1])
   # print( f"Hardware Counts for lts__t_sectors_op_read:", hw_counts[2])
   # print( f"Hardware Counts for lts__t_sectors_op_write:", hw_counts[3])
    print("\033[0;32mPASSED\033[0m")
else:
    print("\033[0;31mFAILED\033[0m");
