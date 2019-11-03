import numpy as np
import scipy.io
import cv2
import nms
import time
import os
import detect_util

from xfdnn.rt import xdnn, xdnn_io

# Our custom FPGA One-shot layer
class XFDNNPyAPI():

  # Called once when the network is initialized
  def __init__(self, param_str):
    self.param_dict = eval(param_str) # Get args from prototxt
    self._args = xdnn_io.make_dict_args(self.param_dict)
    self._numPE = self._args["batch_sz"] # Bryan hack to detremine number of PEs in FPGA

    # Establish FPGA Communication, Load bitstream
    ret, handles = xdnn.createHandle(self._args["xclbin"], "kernelSxdnn_0")
    if ret != 0:
      raise Exception("Failed to open FPGA handle.")
    
    self._args["scaleB"] = 1    
    self._args["PE"] = -1    
    self._streamIds = [0,1,2,3,4,5,6,7] # Allow 8 streams 
    
    # Instantiate runtime interface object
    self.fpgaRT = xdnn.XDNNFPGAOp(handles, self._args)
    self._indictnames = self._args["input_names"]
    self._outdictnames =  self._args["output_names"]
    self._parser = xdnn.CompilerJsonParser(self._args["netcfg"])

  # Called for every batch
  def forward(self, bottom):
    top = [0]*len(self._outdictnames)
    
    indict = {}
    outdict = {}
    
    for i,n in enumerate(self._indictnames):
      # default to 1 batch
      indict[ n ] = np.ascontiguousarray(np.expand_dims(bottom[i], 0))
      
    for i,name in enumerate(self._outdictnames):        
      dim = self._parser.getOutputs()[ name ]
      top[i] = np.empty(dim,dtype=np.float32)
      outdict[ name ] = top[i]

    # Get a free stream if available
    if self._streamIds:
      streamId = self._streamIds.pop(0)
    else:
      return None

    start_time = time.time()
    #self.fpgaRT.execute(indict, outdict, streamId)
    self.fpgaRT.exec_async(indict, outdict, streamId)
    self.fpgaRT.get_result(streamId)
    end_time = time.time()

    self._streamIds.append(streamId) # Return stream

    return outdict


class Detect(object):
  def __init__(self):
    self.expand_scale_=0.0
    self.force_gray_=False
    self.input_mean_value_=128.0
    self.input_scale_=1.0
    self.pixel_blob_name_='pixel-prob'
    self.bb_blob_name_='bb-output-tiled'
    
    self.res_stride_=4
    self.det_threshold_=0.7
    self.nms_threshold_=0.3
    self.input_channels_=3

  def model_init(self,model_path,def_path):

    MLSUITE_ROOT = os.getenv("MLSUITE_ROOT","/opt/ml-suite")
    MLSUITE_PLATFORM = os.getenv("MLSUITE_PLATFORM","alveo-u200")

    param_str = "{\'batch_sz\': 1," +\
                "\'outtrainproto\': None," +\
                "\'input_names\': [u\'data\']," +\
                "\'cutAfter\': \'data\'," +\
                "\'outproto\': \'xfdnn_deploy.prototxt\'," +\
                "\'xdnnv3\': True," +\
                "\'inproto\': \'deploy.prototxt\'," +\
                "\'profile\': False," +\
                "\'trainproto\': None," +\
                "\'weights\': \'deploy.caffemodel_data.h5\'," +\
                "\'netcfg\': \'deploy.compiler.json\'," +\
                "\'quantizecfg\': \'deploy.compiler_quant.json\'," +\
                "\'xclbin\': \'" + MLSUITE_ROOT + "/overlaybins/" + MLSUITE_PLATFORM + "/overlay_4.xclbin\'," +\
                "\'output_names\': [u\'pixel-conv\', u\'bb-output\']," +\
                "\'overlaycfg\': {u\'XDNN_NUM_KERNELS\': u\'2\', u\'SDX_VERSION\': u\'2018.2\', u\'XDNN_VERSION_MINOR\': u\'0\', u\'XDNN_SLR_IDX\': u\'1, 1\', u\'XDNN_DDR_BANK\': u\'0, 3\', u\'XDNN_CSR_BASE\': u\'0x1800000, 0x1810000\', u\'XDNN_BITWIDTH\': u\'8\', u\'DSA_VERSION\': u\'xilinx_u200_xdma_201820_1\', u\'XDNN_VERSION_MAJOR\': u\'3\'}}"

    self.xfdnn_graph_ = XFDNNPyAPI(param_str)

    

  def detect(self,image):

    # transpose HWC (0,1,2) to CHW (2,0,1)
    transformed_image = np.transpose(image,(2,0,1))

    transformed_image=(transformed_image-self.input_mean_value_)*self.input_scale_
    sz=(512,320)
    #sz=image.shape

    # Call FPGA
    output = self.xfdnn_graph_.forward([transformed_image.astype(np.float32)])
      
    # Put CPU layers into postprocess
    pixel_conv = output['pixel-conv']
    pixel_conv_tiled = detect_util.GSTilingLayer_forward(pixel_conv, 8)
    prob = detect_util.SoftmaxLayer_forward(pixel_conv_tiled)
    prob = prob[0,1,...]
    
    bb = output['bb-output']
    bb = detect_util.GSTilingLayer_forward(bb, 8)
    bb = bb[0, ...]
      
    ##import pdb; pdb.set_trace()
    gy = np.arange(0, sz[0], self.res_stride_)
    gx = np.arange(0, sz[1], self.res_stride_)
    gy = gy[0 : bb.shape[1]]
    gx = gx[0 : bb.shape[2]]
    [x, y] = np.meshgrid(gx, gy)
    
    #print bb.shape[1],len(gy),sz[0],sz[1]
    bb[0, :, :] += x
    bb[2, :, :] += x
    bb[1, :, :] += y
    bb[3, :, :] += y
    bb = np.reshape(bb, (4, -1)).T
    prob = np.reshape(prob, (-1, 1))
    bb = bb[prob.ravel() > self.det_threshold_, :]
    prob = prob[prob.ravel() > self.det_threshold_, :]
    rects = np.hstack((bb, prob))
    keep = self.nms(rects, self.nms_threshold_)	
    rects = rects[keep, :]
    rects_expand=[]
    for rect in rects:
      rect_expand=[]
      rect_w=rect[2]-rect[0]
      rect_h=rect[3]-rect[1]
      rect_expand.append(int(max(0,rect[0]-rect_w*self.expand_scale_)))
      rect_expand.append(int(max(0,rect[1]-rect_h*self.expand_scale_)))
      rect_expand.append(int(min(sz[1],rect[2]+rect_w*self.expand_scale_)))
      rect_expand.append(int(min(sz[0],rect[3]+rect_h*self.expand_scale_)))
      rects_expand.append(rect_expand)
 
    return rects_expand 

  def nms(self,dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
      i = order[0]
      keep.append(i)
      xx1 = np.maximum(x1[i], x1[order[1:]])
      yy1 = np.maximum(y1[i], y1[order[1:]])
      xx2 = np.minimum(x2[i], x2[order[1:]])
      yy2 = np.minimum(y2[i], y2[order[1:]])

      w = np.maximum(0.0, xx2 - xx1 + 1)
      h = np.maximum(0.0, yy2 - yy1 + 1)
      inter = w * h
      ovr = inter / (areas[i] + areas[order[1:]] - inter)

      inds = np.where(ovr <= thresh)[0]
      order = order[inds + 1]

    return keep
