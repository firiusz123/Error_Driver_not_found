import onnx, os

folder = r'/Users/hubertm/Documents/College/Events/VASC/Error_Driver_not_found/'
src    = os.path.join(folder, 'drowsiness_matlab.onnx')
dst    = os.path.join(folder, 'drowsiness_matlab_merged.onnx')

print('Loading...')
model = onnx.load(src)   # loads both .onnx and .onnx.data automatically

print('Saving as single file...')
onnx.save(model, dst, save_as_external_data=False)

print(f'Done: {dst}')
print(f'Size: {os.path.getsize(dst)/1024/1024:.1f} MB')

