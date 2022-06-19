using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

using Microsoft.ML.OnnxRuntime;
using Microsoft.MixedReality.WebRTC;
using OpenCvSharp;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp.Dnn;
using System.Linq;
using System.IO;

namespace WebRTCNetCoreSandbox
{
    class Program
    {
        const bool DEBUG = false;

        // https://nietras.com/2021/01/25/onnxruntime/
        // https://www.onnxruntime.ai/  
        // ~6-8 ms for preparation of frames prior to inference
        // Test configs:
        // [Windows, C#, X64], NVIDIA RTX 3090 GPU, AMD 3900x CPU
        // prep and inference with effnet_b0_512 (FP32, no optimizations)
        
        // Providers:
        // CPU: < 10 seconds for model load; ~175 ms (w Unity running)
        // CUDA: ~1 min for model load; ~40 ms (w Unity running)
        // DirectML: - not using GPU for some reason...; ? ms
        // TensorRT: ~10 min for model load; ~ 16 ms (w Unity running)
        // https://developer.nvidia.com/blog/using-windows-ml-onnx-and-nvidia-tensor-cores/

        static async Task Main(string[] args)
        {
            // Update the file path to point to the location of the onnx-models/ folder
            const string basePath = "C:/git/public/hmd-ego-pose/pytorch-sandbox/onnx-models/";

            // Allocate the matrix for frame bytes
            using Mat anchors = CvMatFromFloatArray(LoadTextFileToFloatArray(basePath + "anchors_256.txt"), 4);
            using Mat translationAnchors = CvMatFromFloatArray(LoadTextFileToFloatArray(basePath + "translation_anchors_256.txt"), 3);

            // Set the camera parameters for the desktop webcam or the HoloLens 2 camera
            using Mat cameraParams = CvMatFromFloatArray(LoadTextFileToFloatArray(basePath + "camera_params.txt"), 6);
            Console.WriteLine($"cameraParams: [{cameraParams.At<float>(0)}, {cameraParams.At<float>(1)}, {cameraParams.At<float>(0)}, {cameraParams.At<float>(2)}, {cameraParams.At<float>(3)}, {cameraParams.At<float>(4)}, {cameraParams.At<float>(5)}]");

            // Load and create the onnx model
            const string ModelFilePath = basePath + "model.onnx";
            const int TargetSize = 256;
            const int Channels = 3;

            // Create GPU session 
            int cpuDeviceId = 1; // The CPU device ID to execute on
            int gpuDeviceId = 0; // The GPU device ID to execute on
            var sessionOptions = new SessionOptions();

            // Uncomment specified provider, will need to select the relevant NuGet package for support
            // CPU
            //sessionOptions.AppendExecutionProvider_CPU(cpuDeviceId);

            // CUDA
            sessionOptions.AppendExecutionProvider_CUDA(gpuDeviceId);

            // DirectML                                                                                     
            //sessionOptions.AppendExecutionProvider_DML(gpuDeviceId);

            // TensorRT
            //sessionOptions.AppendExecutionProvider_Tensorrt(gpuDeviceId);
            //sessionOptions.AppendExecutionProvider_CUDA(gpuDeviceId);

            // Optimize
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            using var Session = new InferenceSession(
                ModelFilePath,
                sessionOptions);
            Console.WriteLine($"Successfully loaded onnx model from path: {ModelFilePath}.\n");
            
            const int PoseDataChannelID = 12;
            const string PoseDataChannelName = "pose";
            DataChannel poseDataChannel = null;
            NodeDssSignaler signaler = null;

            AudioTrackSource microphoneSource = null;
            VideoTrackSource webcamSource = null;

            Transceiver audioTransceiver = null;
            Transceiver videoTransceiver = null;
            
            LocalAudioTrack localAudioTrack = null;
            LocalVideoTrack localVideoTrack = null;

            RemoteAudioTrack remoteAudioTrack = null;
            RemoteVideoTrack remoteVideoTrack = null;

            try
            {
                // Create the peer connection
                using var peerConnection = new PeerConnection();

                // Specify some options to configure a STUN server
                var config = new PeerConnectionConfiguration();
                await peerConnection.InitializeAsync(config);
                Console.WriteLine("Peer connection initialized.\n");

                peerConnection.Connected += () => { Console.WriteLine("PeerConnection: connected.\n"); };
                peerConnection.DataChannelAdded += (DataChannel channel) =>
                {
                    poseDataChannel = channel;
                    Console.WriteLine($"Added data channel: '{channel.Label}' (#{channel.ID}).");
                };
                peerConnection.DataChannelRemoved += (DataChannel channel) =>
                {
                    poseDataChannel = null;
                    Console.WriteLine($"Removed data channel: '{channel.Label}' (#{channel.ID}).");
                };
                peerConnection.IceStateChanged += (IceConnectionState newState) => { Console.WriteLine($"ICE state: {newState}\n"); };

                peerConnection.VideoTrackAdded += async (RemoteVideoTrack track) => 
                {

                    // Utility used for evaluating characteristics of the connection
                    // between the HoloLens 2 and PC (latency, available bandwidth)
                    //GetSimpleStats(peerConnection);

                    track.I420AVideoFrameReady += (I420AVideoFrame frame) =>
                    {
                        var watchPrep = new Stopwatch();
                        var watchInference = new Stopwatch();

                        if (DEBUG)
                            watchPrep.Start();

                        // Compute the byte size of I420
                        uint cols = frame.width;
                        uint rows = frame.height;
                        uint pixelSize = cols * rows;
                        uint byteSize = (pixelSize / 2 * 3); // I420 = 12 bits per pixel

                        // Allocate bytes for the frame and copy
                        var bytesNv12 = new byte[byteSize];
                        frame.CopyTo(bytesNv12);

                        // Allocate the matrix for frame bytes
                        using Mat cvMatNv12 = new Mat(
                            (int)(rows + (rows / 2.0f)),
                            (int)cols,
                            MatType.CV_8UC1);
                        Marshal.Copy(bytesNv12, 0, cvMatNv12.Data, (int)byteSize);
                        if (DEBUG)
                            Console.WriteLine($"cvMatNv12: {cvMatNv12.Width} x {cvMatNv12.Height}");

                        //Cv2.ImShow("cvMatNv12 frame", cvMatNv12);
                        //Cv2.WaitKey(1);

                        // Convert to BGR from YV12
                        using Mat cvMatBgr = new Mat();
                        Cv2.CvtColor(cvMatNv12, cvMatBgr, ColorConversionCodes.YUV2BGR_YV12);
                        if (DEBUG)
                            Console.WriteLine($"cvMatBgr: {cvMatBgr.Width} x {cvMatBgr.Height}");

                        //Cv2.ImShow("cvMatBgr frame", cvMatBgr);
                        //Cv2.WaitKey(1);

                        // Center crop and rescale the image to ensure that the
                        // input data represents a similar view as the training data 
                        // (centered on the surgical drill and zoomed in)
                        int crop_size = 256;
                        int resized_width = 512;
                        int resized_height = 512;
                        var cvMatBgrResize = CenterCropAndRescaleMat(crop_size, resized_width, resized_height, cvMatBgr);

                        //Cv2.ImShow("cvMatBgrResize frame", cvMatBgrResize);
                        //Cv2.WaitKey(1);

                        // Preprocess the image
                        var normMean = new Scalar(0.485f, 0.456f, 0.406f);
                        var normStd = new Scalar(0.229f, 0.224f, 0.225f);

                        var resizeResult = ResizeAndNormalizeMat(TargetSize, cvMatBgrResize, normMean, normStd);
                        using Mat cvMatBgrProc = resizeResult.Item1;
                        float resizeScale = resizeResult.Item2;

                        if (DEBUG)
                            Console.WriteLine($"cvMatBgrProc: {cvMatBgrProc.Width} x {cvMatBgrProc.Height}");

                        //Cv2.ImShow("Processed frame", cvMatBgrProc);
                        //Cv2.WaitKey(1);

                        // Create a blob from the frame
                        var TargetImageSize = new OpenCvSharp.Size(TargetSize, TargetSize);
                        float scale = 1.0f;
                        using Mat cvMatBgrProcCHW = CvDnn.BlobFromImage(cvMatBgrProc, scale, TargetImageSize, new Scalar(), false, false);

                        // Allocate float[] of size and copy data from opencv mat
                        var inputByteSize = 1 * Channels * TargetSize * TargetSize;
                        var inputFloatArr = new float[inputByteSize];
                        Marshal.Copy(cvMatBgrProcCHW.Data, inputFloatArr, 0, inputByteSize);

                        if (DEBUG)
                        {
                            watchPrep.Stop();
                            Console.WriteLine($"Image preprocessing in {watchPrep.ElapsedMilliseconds} ms");
                        }

                        watchInference.Start();

                        // Create the model inputs from and copy from opencv mat (1, 3, 224, 224)
                        var inputShape = new int[] { 1, 3, TargetSize, TargetSize };
                        var inputTensor = new DenseTensor<float>(new Memory<float>(inputFloatArr), inputShape);
                        var inputOnnxValues = new List<NamedOnnxValue>
                        {
                            NamedOnnxValue.CreateFromTensor("input", inputTensor)
                        };

                        // Create session and run inference
                        using var results = Session.Run(inputOnnxValues);

                        // Create the results tensors]
                        var resultsArray = results.ToArray();
                        var dim_1 = resultsArray[5].AsEnumerable<float>().ToArray().Length;

                        using Mat regression = CvMatFromFloatArray(resultsArray[5].AsEnumerable<float>().ToArray(), 4);
                        using Mat classification = CvMatFromFloatArray(resultsArray[6].AsEnumerable<float>().ToArray(), 1);
                        using Mat rotation = CvMatFromFloatArray(resultsArray[7].AsEnumerable<float>().ToArray(), 3);
                        using Mat translation_raw = CvMatFromFloatArray(resultsArray[8].AsEnumerable<float>().ToArray(), 3);
                        using Mat hand = CvMatFromFloatArray(resultsArray[9].AsEnumerable<float>().ToArray(), 3);

                        if (DEBUG)
                        {
                            Console.WriteLine($"regression: [{regression.At<float>(0)}, {regression.At<float>(1)}, {regression.At<float>(2)}, {regression.At<float>(3)}]");
                            Console.WriteLine($"classification: [{classification.At<float>(0)}]");
                            Console.WriteLine($"rotation: [{rotation.At<float>(0)}, {rotation.At<float>(1)}, {rotation.At<float>(2)}]");
                            Console.WriteLine($"translation_raw: [{translation_raw.At<float>(0)}, {translation_raw.At<float>(1)}, {translation_raw.At<float>(2)}]");
                            Console.WriteLine($"hand: [{hand.At<float>(0)}, {hand.At<float>(1)}, {hand.At<float>(2)}]");

                            Console.WriteLine($"regression: {regression.Width} x {regression.Height}");
                            Console.WriteLine($"classification: {classification.Width} x {classification.Height}");
                            Console.WriteLine($"rotation: {rotation.Width} x {rotation.Height}");
                            Console.WriteLine($"translation_raw: {translation_raw.Width} x {translation_raw.Height}");
                            Console.WriteLine($"hand: {hand.Width} x {hand.Height}");
                        }

                        // STAGE 1: format translation data
                        using Mat translation = format_translation(
                            translationAnchors, translation_raw, cameraParams);

                        // STAGE 2: format and clip bounding box data
                        var result = format_bboxes(
                            new OpenCvSharp.Size(TargetSize, TargetSize),
                            anchors, regression);

                        using Mat xmin = result.Item1;
                        using Mat ymin = result.Item2;
                        using Mat xmax = result.Item3;
                        using Mat ymax = result.Item4;

                        // STAGE 3: filter detections using confidence threshold and non-maximum suppression
                        var filteredResult = filter_detections(
                            xmin, ymin, xmax, ymax,
                            classification,
                            rotation,
                            translation);

                        float residualScore = filteredResult.Item1;
                        Rect residualBox = filteredResult.Item2;
                        Point3f residualRotation = filteredResult.Item3;
                        Point3f residualTranslation = filteredResult.Item4;

                        //Cv2.ImShow("cvMatNv12 frame", cvMatNv12);
                        //Cv2.WaitKey(1);

                        watchInference.Stop();
                        Console.WriteLine($"Model inference in {watchInference.ElapsedMilliseconds} ms");

                        // Copy the transform into new float array
                        var floatArray = new float[] {
                            residualRotation.X, residualRotation.Y, residualRotation.Z,
                            residualTranslation.X, residualTranslation.Y, residualTranslation.Z
                        };

                        // Create a byte array and copy the floats 
                        var byteArray = new byte[floatArray.Length * 4];
                        Buffer.BlockCopy(floatArray, 0, byteArray, 0, byteArray.Length);

                        // Send the byte array over the channel if the channel is available
                        if (poseDataChannel.State == DataChannel.ChannelState.Open)
                        {
                            poseDataChannel.SendMessage(byteArray);

                            if (DEBUG)
                            {
                                Console.WriteLine($"Sent {byteArray.Length} byte packet over data channel.");
                            }
                        }
                    };
                };

                // It is CRUCIAL to add any data channel BEFORE the SDP offer is sent, if data channels are
                // to be used at all. Otherwise the SCTP will not be negotiated, and then all channels will
                // stay forever in the kConnecting state.
                // https://stackoverflow.com/questions/43788872/how-are-data-channels-negotiated-between-two-peers-with-webrtc
                var newDataChannel = await peerConnection.AddDataChannelAsync(
                    PoseDataChannelID,
                    PoseDataChannelName,
                    true, true);

                peerConnection.LocalSdpReadytoSend += (SdpMessage message) =>
                {
                    var msg = NodeDssSignaler.Message.FromSdpMessage(message);
                    signaler.SendMessageAsync(msg);

                };
                peerConnection.IceCandidateReadytoSend += (IceCandidate iceCandidate) =>
                {
                    var msg = NodeDssSignaler.Message.FromIceCandidate(iceCandidate);
                    signaler.SendMessageAsync(msg);
                };

                peerConnection.Connected += () => {Console.WriteLine("PeerConnection: connected.");};
                peerConnection.IceStateChanged += (IceConnectionState newState) => {Console.WriteLine($"ICE state: {newState}");};

                // Initialize the signaler
                signaler = new NodeDssSignaler()
                {
                    HttpServerAddress = "http://192.168.2.29:3000/",
                    LocalPeerId = "WebRTCSandbox",
                    RemotePeerId = "HoloLens2MachineLearningWebRTC",
                };
                signaler.OnMessage += async (NodeDssSignaler.Message msg) =>
                {
                    switch (msg.MessageType)
                    {
                        case NodeDssSignaler.Message.WireMessageType.Offer:
                            // Wait for the offer to be applied
                            await peerConnection.SetRemoteDescriptionAsync(msg.ToSdpMessage());
                            // Once applied, create an answer
                            peerConnection.CreateAnswer();
                            break;

                        case NodeDssSignaler.Message.WireMessageType.Answer:
                            // No need to await this call; we have nothing to do after it
                            peerConnection.SetRemoteDescriptionAsync(msg.ToSdpMessage());
                            break;

                        case NodeDssSignaler.Message.WireMessageType.Ice:
                            peerConnection.AddIceCandidate(msg.ToIceCandidate());
                            break;
                    }
                };
                signaler.StartPollingAsync();

                if (!peerConnection.IsConnected)
                {
                    audioTransceiver = peerConnection.AddTransceiver(MediaKind.Audio);
                    videoTransceiver = peerConnection.AddTransceiver(MediaKind.Video);

                    peerConnection.CreateOffer();
                    Console.WriteLine("Created offer for peer connection.\n");
                }

                // Wait for the user to terminate the application
                Console.WriteLine("Press a key to terminate the application...");
                Console.ReadKey(true);
                signaler.StopPollingAsync();
                Console.WriteLine("Program terminated.");
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }

            localAudioTrack?.Dispose();
            localVideoTrack?.Dispose();
            microphoneSource?.Dispose();
            webcamSource?.Dispose();
        }

        static Mat CenterCropAndRescaleMat(int crop_size, int resized_width, int resized_height, Mat src)
        {
            // Compute the width and height offsets
            int offsetW = (src.Cols - crop_size) / 2;
            int offsetH = (src.Rows - crop_size) / 2;

            // Create the region of interest at center of image
            Rect roi = new Rect(offsetW, offsetH, crop_size, crop_size);
            Mat dst = new Mat(src, roi);

            // Increase image size
            Cv2.Resize(dst, dst, new OpenCvSharp.Size(resized_width, resized_height));

            return dst;
        }

        static (Mat, float) ResizeAndNormalizeMat(
            int img_size, Mat src,
            Scalar normMean, Scalar normStd)
        {
            int image_height = src.Rows;
            int image_width = src.Cols;

            float scale = 0;
            int resized_height, resized_width;

            if (image_height > image_width)
            {
                scale = (float)img_size / image_height;
                resized_height = img_size;
                resized_width = (int)(image_width * scale);
            }
            else
            {
                scale = (float)img_size / image_width;
                resized_height = (int)(image_height * scale);
                resized_width = img_size;
            }

            // Resize image
            Mat dst = new Mat();
            Cv2.Resize(src, dst, new OpenCvSharp.Size(resized_width, resized_height));

            // Convert the matrix format to avoid rounding errors
            dst.ConvertTo(dst, MatType.CV_32F);

            // Transform the image (copy the ToTensor behaviour)
            Cv2.Divide(dst, 255.0f, dst);

            // Normalize with parameters
            Cv2.Subtract(dst, normMean, dst);
            Cv2.Divide(dst, normStd, dst);

            int padH = (int)(img_size - resized_height);
            int padW = (int)(img_size - resized_width);

            Cv2.CopyMakeBorder(
                dst,
                dst,
                0, padH,
                0, padW,
                BorderTypes.Constant, new Scalar(0));

            return (dst, scale);
        }

        //static async void GetSimpleStats(PeerConnection pc)
        //{
        //    var sp = await pc.GetSimpleStatsAsync();
        //    IEnumerable<PeerConnection.TransportStats> transStats = sp.GetStats<PeerConnection.TransportStats>();
        //    ulong v1, v2;
        //    long v3;
        //    using (var sequenceEnum = transStats.GetEnumerator())
        //    {
        //        while (sequenceEnum.MoveNext())
        //        {
        //            v1 = sequenceEnum.Current.BytesReceived;
        //            v2 = sequenceEnum.Current.BytesSent;
        //            v3 = sequenceEnum.Current.TimestampUs;

        //            DateTime dt = ToDateTimeForEpochMSec(v3);

        //            Console.WriteLine($"Transport: bytes received: {v1} bytes sent: {v2} time: {dt.Millisecond} msecs");
        //        }
        //    }
        //}

        static float[] LoadTextFileToFloatArray(string f)
        {
            TextReader textReader = File.OpenText(f);
            string line = textReader.ReadLine();
            string[] bits = line.Split(' ');
            float[] parsed = Array.ConvertAll(line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries), float.Parse);
            Console.WriteLine($"Successfully loaded data from: {f}");
            return parsed;
        }

        static Mat CvMatFromFloatArray(float[] fa, int dim_2)
        {
            var dim_1 = fa.Length / dim_2;
            //Mat m = new Mat(dim_1, dim_2, MatType.CV_32F, fa);
            Mat m = new Mat(dim_1, dim_2, MatType.CV_32F);
            Marshal.Copy(fa, 0, m.Data, dim_1 * dim_2);
            //Console.WriteLine($"Float array to opencv mat. Mat shape: {m.Size()}");
            return m.Transpose();
        }

        static Mat format_translation(
            Mat translation_anchors,
            Mat translation_raw,
            Mat camera_parameters_input)
        {
            var regress = regress_translation(translation_anchors, translation_raw);
            Mat x = regress.Item1;
            Mat y = regress.Item2;
            Mat Tz = regress.Item3;

            var translation = calculate_txty(
                x, y, Tz,
                camera_parameters_input.At<float>(0), camera_parameters_input.At<float>(1),
                camera_parameters_input.At<float>(2), camera_parameters_input.At<float>(3),
                camera_parameters_input.At<float>(4), camera_parameters_input.At<float>(5));

            return translation;
        }

        // https://stackoverflow.com/questions/65623993/how-to-convert-16-digit-epoch-timestamp-to-datetime-in-c-sharp-without-losing-mi
        static DateTime ToDateTimeForEpochMSec(double microseconds)
        {
            var epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
            long ticksPerMicrosecond = TimeSpan.TicksPerMillisecond / 1000;
            long ticks = (long)(microseconds * ticksPerMicrosecond);
            DateTime tempDate = epoch.AddTicks(ticks);
            return tempDate;
        }

        static (Mat, Mat, Mat) regress_translation(
            Mat translation_anchors,
            Mat deltas)
        {
            // Regress translation
            // (49104, 3) -> (49104)
            // stride = translation_anchors[:, :, -1]
            OpenCvSharp.Range[] r1 = new OpenCvSharp.Range[]
            {
                new OpenCvSharp.Range(2, 3),
                OpenCvSharp.Range.All
            };
            Mat stride = translation_anchors[r1];
            if (DEBUG)
                Console.WriteLine($"stride: {stride.Size()}");

            // translation_anchors_0 = translation_anchors[:, :, 0]
            // deltas_0 = deltas[:, :, 0]
            OpenCvSharp.Range[] r2 = new OpenCvSharp.Range[]
            {
                new OpenCvSharp.Range(0, 1),
                OpenCvSharp.Range.All
            };
            Mat translation_anchors_0 = translation_anchors[r2];
            Mat deltas_0 = deltas[r2];
            if (DEBUG)
            {
                Console.WriteLine($"translation_anchors_0: {translation_anchors_0.Size()}");
                Console.WriteLine($"deltas_0: {deltas_0.Size()}");

            }

            // translation_anchors_1 = translation_anchors[:, :, 1]
            // deltas_1 = deltas[:, :, 1]
            OpenCvSharp.Range[] r3 = new OpenCvSharp.Range[]
            {
                new OpenCvSharp.Range(1, 2),
                OpenCvSharp.Range.All
            };
            Mat translation_anchors_1 = translation_anchors[r3];
            Mat deltas_1 = deltas[r3];

            if (DEBUG)
            {
                Console.WriteLine($"translation_anchors_1: {translation_anchors_0.Size()}");
                Console.WriteLine($"deltas_1: {deltas_0.Size()}");
            }

            Mat x = translation_anchors_0 + (deltas_0.Mul(stride));
            Mat y = translation_anchors_1 + (deltas_1.Mul(stride));

            if (DEBUG)
            {
                Console.WriteLine($"x: {x.Size()}");
                Console.WriteLine($"y: {y.Size()}");
            }

            // Tz = deltas[:, :, 2]
            OpenCvSharp.Range[] r4 = new OpenCvSharp.Range[]
            {
                new OpenCvSharp.Range(2, 3),
                OpenCvSharp.Range.All
            };
            Mat Tz = deltas[r4];
            
            if (DEBUG)
                Console.WriteLine($"Tz: {Tz.Size()}");

            return (x, y, Tz);
        }

        static Mat calculate_txty(
            Mat x, Mat y, Mat Tz,
            float fx, float fy,
            float px, float py,
            float tz_scale,
            float image_scale)
        {
            if (DEBUG)
            {
                Console.WriteLine($"x: {x.Size()}");
                Console.WriteLine($"y: {y.Size()}");
                Console.WriteLine($"Tz: {Tz.Size()}");

                Console.WriteLine($"fx: {fx} fy: {fy}");
                Console.WriteLine($"px: {px} py: {py}");
                Console.WriteLine($"tz_scale: {tz_scale} image_scale: {image_scale}");
            }

            x = x / image_scale;
            y = y / image_scale;
            Mat tz = Tz * tz_scale;

            x = x - px;
            y = y - py;

            Mat tx = (x.Mul(tz)) / fx;
            Mat ty = (y.Mul(tz)) / fy;

            if (DEBUG)
            {
                Console.WriteLine($"tx: {tx.Size()}");
                Console.WriteLine($"ty: {ty.Size()}");
                Console.WriteLine($"tz: {tz.Size()}");
            }

            Mat translation = new Mat();
            List<Mat> matrices = new List<Mat> { tx, ty, tz };
            Cv2.VConcat(matrices, translation);
            
            if (DEBUG)
                Console.WriteLine($"translation: {translation.Size()}");

            return translation;
        }

        static (Mat, Mat, Mat, Mat) format_bboxes(
            OpenCvSharp.Size image_size,
            Mat anchors,
            Mat bbox_regression)
        {
            var result = regress_boxes(
                anchors,
                bbox_regression);

            Mat xmin = result.Item1;
            Mat ymin = result.Item2;
            Mat xmax = result.Item3;
            Mat ymax = result.Item4;

            var result2 = clip_boxes(
                image_size.Width, image_size.Height,
                xmin, ymin, xmax, ymax);

            return (result2.Item1, result2.Item2, result2.Item3, result2.Item4);
        }

        static (Mat, Mat, Mat, Mat) regress_boxes(
            Mat boxes,
            Mat deltas)
        {
            if (DEBUG)
            {
                Console.WriteLine($"boxes: {boxes.Size()}");
                Console.WriteLine($"deltas: {deltas.Size()}");
            }

            // (49104, 3) -> (49104)
            // boxes[..., 0]
            OpenCvSharp.Range[] r1 = new OpenCvSharp.Range[]
            {
                new OpenCvSharp.Range(0, 1),
                OpenCvSharp.Range.All
            };
            Mat boxes_0 = boxes[r1];
            Mat tx = deltas[r1];
            
            if (DEBUG)
            {
                Console.WriteLine($"boxes_0: {boxes_0.Size()}");
                Console.WriteLine($"tx: {tx.Size()}");
            }

            // boxes[..., 1]
            OpenCvSharp.Range[] r2 = new OpenCvSharp.Range[]
            {
                new OpenCvSharp.Range(1, 2),
                OpenCvSharp.Range.All
            };
            Mat boxes_1 = boxes[r2];
            Mat ty = deltas[r2];

            if (DEBUG)
            {
                Console.WriteLine($"boxes_1: {boxes_1.Size()}");
                Console.WriteLine($"ty: {ty.Size()}");
            }

            // boxes[..., 2]
            OpenCvSharp.Range[] r3 = new OpenCvSharp.Range[]
            {
                new OpenCvSharp.Range(2, 3),
                OpenCvSharp.Range.All
            };
            Mat boxes_2 = boxes[r3];
            Mat th = deltas[r3];

            if (DEBUG)
            {
                Console.WriteLine($"boxes_2: {boxes_2.Size()}");
                Console.WriteLine($"th: {th.Size()}");
            }

            // boxes[..., 3]
            OpenCvSharp.Range[] r4 = new OpenCvSharp.Range[]
            {
                new OpenCvSharp.Range(3, 4),
                OpenCvSharp.Range.All
            };
            Mat boxes_3 = boxes[r4];
            Mat tw = deltas[r4];

            if (DEBUG)
            {
                Console.WriteLine($"boxes_3: {boxes_3.Size()}");
                Console.WriteLine($"tw: {tw.Size()}");
            }

            Mat cxa = (boxes_0 + boxes_2) / 2;
            Mat cya = (boxes_1 + boxes_3) / 2;
            Mat wa = boxes_2 - boxes_0;
            Mat ha = boxes_3 - boxes_1;

            Mat w = new Mat();
            Mat h = new Mat();
            Cv2.Exp(tw, w);
            Cv2.Exp(th, h);

            w = w.Mul(wa);
            h = h.Mul(ha);

            Mat cy = ty.Mul(ha) + cya;
            Mat cx = tx.Mul(wa) + cxa;

            Mat ymin = cy - h / 2;
            Mat xmin = cx - w / 2;
            Mat ymax = cy + h / 2;
            Mat xmax = cx + w / 2;

            if (DEBUG)
            {
                Console.WriteLine($"ymin: {ymin.Size()}");
                Console.WriteLine($"xmin: {xmin.Size()}");
                Console.WriteLine($"ymax: {ymax.Size()}");
                Console.WriteLine($"xmax: {xmax.Size()}");
            }

            return (xmin, ymin, xmax, ymax);
        }

        // https://stackoverflow.com/questions/7552896/most-efficient-way-to-clamp-values-in-an-opencv-mat
        static Mat clamp(Mat mat, int lowerBound, int upperBound)
        {
            Mat maxMat = new Mat();
            Cv2.Max(mat, lowerBound, maxMat);
            Cv2.Min(maxMat, upperBound, mat);
            return mat;
        }

        static (Mat, Mat, Mat, Mat) clip_boxes(
            int width, int height,
            Mat xmin, Mat ymin, Mat xmax, Mat ymax)
        {
            clamp(xmin, 0, width - 1);
            clamp(ymin, 0, height - 1);
            clamp(xmax, 0, width - 1);
            clamp(ymax, 0, width - 1);

            if (DEBUG)
            {
                Console.WriteLine($"ymin: {ymin.Size()}");
                Console.WriteLine($"xmin: {xmin.Size()}");
                Console.WriteLine($"ymax: {ymax.Size()}");
                Console.WriteLine($"xmax: {xmax.Size()}");
            }

            return (xmin, ymin, xmax, ymax);
        }

        static (float, Rect, Point3f, Point3f) filter_detections(
            Mat xmin, Mat ymin, Mat xmax, Mat ymax,
            Mat scores,
            Mat rotation_input,
            Mat translation)
        {
            // xmax: 49104 x 1
            // classification: 1 x 49104 x 1
            // rotation : 1 x 49104 x 3
            // translation : 1 x 49104 x 3

            // Define control parameters
            float nms_threshold = 0.5f;
            float score_threshold = 0.5f;
            int top_K = 10;

            if (DEBUG)
            {
                Console.WriteLine($"scores: {scores.Size()}");
                Console.WriteLine($"rotation_input: {rotation_input.Size()}");
                Console.WriteLine($"translation: {translation.Size()}");
            }

            List<int> indices = new List<int>();
            if (DEBUG)
                Console.WriteLine($"scores.Size(): {scores.Size()}");
            for (int i = 0; i < scores.Size().Width; i++)
            {
                if (scores.At<float>(0, i) > score_threshold)
                {
                    indices.Add(i);
                    if (DEBUG)
                        Console.WriteLine($"indices.Add(i): val: {scores.Get<float>(0, i)} index: {i}");
                }
            }

            // Apply the indices to the predictions to filter
            // loop over indices, copy the data from vec[index] column to subData[index]
            // for the specified indices
            //cv::Mat xmin_filt, ymin_filt, xmax_filt, ymax_filt, scores_thr_filt;
            List<float> filtered_scores = new List<float>();
            List<Rect> filtered_boxes = new List<Rect>();
            List<Point3f> filtered_rotation = new List<Point3f>();
            List<Point3f> filtered_translation = new List<Point3f>();

            for (int index = 0; index < indices.Count; index++)
            {
                // push each row into new matrix
                if (DEBUG)
                    Console.WriteLine($"indices[index]: {indices[index]}");

                // filtered_boxes = boxes[list(indices_.T)]
                // boxes in (x1, y1, x2, y2) format which are converted to
                // (x, y, w, h) format when placed in the OpenCV rect container
                filtered_boxes.Add(
                    new Rect(
                        (int)xmin.Col(indices[index]).At<float>(0),
                        (int)ymin.Col(indices[index]).At<float>(0),
                        (int)xmax.Col(indices[index]).At<float>(0),
                        (int)ymax.Col(indices[index]).At<float>(0)));

                if (DEBUG)
                {
                    Console.WriteLine($"" +
                        $"xmin: {xmin.Col(indices[index]).At<float>(0)} \n" +
                        $"ymin: {ymin.Col(indices[index]).At<float>(0)} \n" +
                        $"xmax: {xmax.Col(indices[index]).At<float>(0)} \n" +
                        $"ymax: {ymax.Col(indices[index]).At<float>(0)}");
                }

                // filtered_scores = scores_[list(indices_.T)]
                //scores_thr_filt.push_back(scores_thr.row(indices[index]));
                //filtered_scores.emplace_back(scores_thr.row(indices[index]).at<float>(0));
                filtered_scores.Add(
                    scores.Col(indices[index]).At<float>(0));

                if (DEBUG)
                {
                    Console.WriteLine($"" +
                        $"scores: {scores.Col(indices[index]).At<float>(0)}");
                }

                filtered_rotation.Add(
                    new Point3f(
                        rotation_input.Col(indices[index]).At<float>(0),
                        rotation_input.Col(indices[index]).At<float>(1),
                        rotation_input.Col(indices[index]).At<float>(2)));

                if (DEBUG)
                {
                    Console.WriteLine($"rotation_input: " +
                        $"{rotation_input.Col(indices[index]).At<float>(0)}, " +
                        $"{rotation_input.Col(indices[index]).At<float>(1)}, " +
                        $"{rotation_input.Col(indices[index]).At<float>(2)}");
                }


                filtered_translation.Add(
                    new Point3f(
                        translation.Col(indices[index]).At<float>(0),
                        translation.Col(indices[index]).At<float>(1),
                        translation.Col(indices[index]).At<float>(2)));

                if (DEBUG)
                {
                    Console.WriteLine($"translation: " +
                        $"{translation.Col(indices[index]).At<float>(0)}, " +
                        $"{translation.Col(indices[index]).At<float>(1)}, " +
                        $"{translation.Col(indices[index]).At<float>(2)}");
                }
            }

            // NMS with filtered boxes, filtered scores, nms_threshold
            // nms_indices = torchvision.ops.nms(filtered_boxes, filtered_scores, nms_threshold)
            int[] residual_indices_int_arr = new int[] { };
            CvDnn.NMSBoxes(filtered_boxes, filtered_scores, score_threshold, nms_threshold, out residual_indices_int_arr, topK: top_K);
            //nms2(filtered_boxes, filtered_scores, residual_boxes, residual_indices, nms_threshold);

            // Iterate across the nms indices and find the highest confidence point
            // use relevant index to select optimal boxes, scores, rotation, and translation data
            int best_index = 0;
            float prior_score = 0f;

            if (DEBUG)
                Console.WriteLine("residual_indices_int_arr: ");
            for (int i = 0; i < residual_indices_int_arr.Length; ++i)
            {
                var current_score = filtered_scores.ElementAt(residual_indices_int_arr[i]);

                if (current_score > prior_score)
                {
                    best_index = residual_indices_int_arr[i];

                    if (DEBUG)
                    {
                        Console.WriteLine(
                            $"best_index: {best_index} \n" +
                            $"current_score: {current_score} \n" +
                            $"prior_score: {prior_score} \n");
                    }
    
                    prior_score = current_score;
                }
                if (DEBUG)
                    Console.WriteLine($"i: {i} residual_indices_int_arr: {residual_indices_int_arr[i]} ");
            }

            float residual_score = 0;
            Rect residual_box = new Rect(0, 0, 0, 0);
            Point3f residual_rotation = new Point3f(0, 0, 0);
            Point3f residual_translation = new Point3f(0, 0, 0);
            if (filtered_scores.Count > 0)
            {
                residual_score = filtered_scores.ElementAt(best_index);
                residual_box = filtered_boxes.ElementAt(best_index);
                residual_rotation = filtered_rotation.ElementAt(best_index);
                residual_translation = filtered_translation.ElementAt(best_index);

                // Adjust rotation by PI
                residual_rotation *= (float)Math.PI;

                // Scale translation from mm -> m
                residual_translation *= (1 / 1000.0f);
            }
            if (DEBUG)
            {
                Console.WriteLine(
                "\n final predictions: \n" +
                $"  residual_score: [{residual_score}] \n" +
                $"  residual_box: [{residual_box.X}, {residual_box.Y}, {residual_box.Width}, {residual_box.Height}] \n" +
                $"  residual_rotation: [{residual_rotation.X}, {residual_rotation.Y}, {residual_rotation.Z}] \n" +
                $"  residual_translation: [{residual_translation.X}, {residual_translation.Y}, {residual_translation.Z}]\n");
            }
            return (residual_score, residual_box, residual_rotation, residual_translation);
        }
    }
}
