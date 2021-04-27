use mmpose frame to export lite_hrnet onnx model


```
#chanes 1: line86
    def get_avgpool_s_k_sz(self,x):
        outputsz = np.array(x[-1].size()[-2:])
        stridesz_list = []
        kernelsz_lsit = []
        for index in range(len(x)):
            inputsz_i = np.array(x[index].size()[-2:])
            stridesz_i = np.floor(inputsz_i / outputsz).astype(np.int32)
            kernelsz_i = inputsz_i - (outputsz - 1) * stridesz_i
            stridesz_list.append(stridesz_i)
            kernelsz_lsit.append(kernelsz_i)
        return stridesz_list,kernelsz_lsit,outputsz

    def forward(self, x):
        # mini_size = x[-1].size()[-2:]
        # out = [F.adaptive_avg_pool2d(s, mini_size) for s in x[:-1]] + [x[-1]]
        # out = torch.cat(out, dim=1)

        stridesz_list, kernelsz_list, outputsz = self.get_avgpool_s_k_sz(x)
        out = [torch.nn.AvgPool2d(kernel_size=kernelsz.tolist(),stride=stridesz.tolist())(s) for s,kernelsz,stridesz in
               zip(x[:-1],kernelsz_list,stridesz_list) ] + [x[-1]]
        out = torch.cat((out), dim=1)

        out = self.conv1(out)
        out = self.conv2(out)
        out = torch.split(out, self.channels, dim=1)
        out = [
            s * F.interpolate(a, size=s.size()[-2:], mode='nearest')
            for s, a in zip(x, out)
        ]


        return out
        
# changes 2:line30

```
        # self.global_avgpool = nn.AvgPool2d(1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
```

when run pytorchonnx.py,warning will happen,just ignore

```
Use load_from_local loader
/home/cody/anaconda3/envs/LiteHRNET/lib/python3.7/site-packages/mmpose/models/backbones/utils/channel_shuffle.py:20: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert (num_channels % groups == 0), ('num_channels should be '
/home/cody/anaconda3/envs/LiteHRNET/lib/python3.7/site-packages/mmpose/models/backbones/litehrnet.py:87: TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  outputsz = np.array(x[-1].size()[-2:])
/home/cody/anaconda3/envs/LiteHRNET/lib/python3.7/site-packages/mmpose/models/backbones/litehrnet.py:91: TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  inputsz_i = np.array(x[index].size()[-2:])
Successfully exported ONNX model: lite_hr.onnx

Process finished with exit code 0

```


'''
