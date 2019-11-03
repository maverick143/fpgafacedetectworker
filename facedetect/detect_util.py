import numpy as np

def GSTilingLayer_forward(bottom, stride):
  stride_sq = stride**2;

  input_batch = bottom.shape[0]
  input_channels = bottom.shape[1]
  input_height = bottom.shape[2]
  input_width = bottom.shape[3]

  output_batch = input_batch
  output_channels = input_channels/stride_sq
  output_height = input_height*stride
  output_width = input_width*stride

  
  top = np.zeros([output_batch, output_channels, output_height, output_width],dtype=np.float32)

  for n in range(input_batch):                 
    for ic in range(input_channels):
      off_div = (ic / output_channels) / stride;
      off_mod = (ic / output_channels) % stride;
      oc = ic % output_channels;
      for iy in range(input_height):          
        oy = iy * stride + off_div;
        ox = off_mod - stride
        top[n,oc,oy,off_mod::stride] = bottom[n,ic,iy,:input_width]
          
  return top


def SoftmaxLayer_forward(bottom):  
  input_batch = bottom.shape[0]
  input_channels = bottom.shape[1]
  input_height = bottom.shape[2]
  input_width = bottom.shape[3]

  top = np.zeros([input_batch, input_channels, input_height, input_width],dtype=np.float32)

  for n in range(input_batch):

    scale_data = np.zeros([input_height,input_width],dtype=np.float32)
    scale_data = bottom[n,0,...]

    for c in range(1,input_channels):
      scale_data = np.maximum(scale_data, bottom[n,c,...])
    
    tmp_bottom = bottom[n,...] - scale_data
    tmp_bottom = np.exp(tmp_bottom)

    scale_data = np.sum(tmp_bottom, axis=0)
    tmp_bottom = tmp_bottom / scale_data
    top[n] = tmp_bottom

  return top
