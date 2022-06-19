using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Microsoft.MixedReality.WebRTC.Unity
{
    public class PoseDataChannel : MonoBehaviour
    {
        public PeerConnection PeerConnection;
        public ushort PoseDataChannelID = 12;
        public string PoseDataChannelName = "pose";
        public GameObject PoseDataGo;
        public bool isDesktop = false;

        public bool IsFlipXTranslation = false;
        public bool IsFlipYTranslation = false;
        public bool IsFlipZTranslation = false;


        // Train: 480, 480 and 256 x 256 images
        public Vector2 centerPointTrainXY;
        public Vector2 resolutionTrainXY;

        // Desktop: 572.41, 573.57, 640 x 480 images
        // HoloLens: 687.70, 688.89, 896 x 504 images
        public Vector2 centerPointTestXY;
        public Vector2 resolutionTestXY;

        public Matrix4x4 CameraToWorldUnity;

        private DataChannel _poseDataChannel;
        private ConcurrentQueue<Action> _mainThreadWorkQueue = new ConcurrentQueue<Action>();

        private void Update()
        {
            // Execute any pending work enqueued by background tasks
            while (_mainThreadWorkQueue.TryDequeue(out Action workload))
            {
                workload();
            }
        }

        /// <summary>
        /// Initialize the WebRTC data channel with ID and string name.
        /// </summary>
        public async void InitializeDataChannel()
        {
            var newDataChannel = await PeerConnection.Peer.AddDataChannelAsync(
                PoseDataChannelID,
                PoseDataChannelName,
                true, true);

            // Add to main thread queue
            _mainThreadWorkQueue.Enqueue(() =>
            {
                _poseDataChannel = newDataChannel;
                Debug.Log($"Added data channel: '{_poseDataChannel.Label}' (#{_poseDataChannel.ID}).");

                _poseDataChannel.MessageReceived += OnDataChannelMessageReceived;
            });
        }

        public void CloseDataChannel()
        {
            _mainThreadWorkQueue.Enqueue(() =>
            {
                Debug.Log($"Removed data channel: '{_poseDataChannel.Label}' (#{_poseDataChannel.ID}).");
                _poseDataChannel = null;
            });
        }

        /// <summary>
        /// When message is received from the client, recieve bytes 
        /// and format into a transformation to apply to the tracked GO
        /// </summary>
        /// <param name="byteArray"></param>
        private void OnDataChannelMessageReceived(byte[] byteArray)
        {
            // Allocate float array to copy bytes into
            var floatArray = new float[byteArray.Length * 4];
            Buffer.BlockCopy(byteArray, 0, floatArray, 0, byteArray.Length);

            //Debug.Log($"Received message in data channel: \n" +
            //    $"Filtered_rotation: [ {floatArray[0]}, {floatArray[1]}, {floatArray[2]}" +
            //    $"Filtered_translation: [{floatArray[3]}, {floatArray[4]}, {floatArray[5]}] \n");

            // Convert float array of length 6 to unity vector3
            var rvec = new Vector3(floatArray[0], floatArray[1], floatArray[2]);
            var tvec = new Vector3(floatArray[3], floatArray[4], floatArray[5]);

            // Handle exploratory flipping to achieve correct pose
            if (IsFlipXTranslation)
                tvec.x *= -1f;
            if (IsFlipYTranslation)
                tvec.y *= -1f;
            if (IsFlipZTranslation)
                tvec.z *= -1f;

            // Compute the quaternion rotation
            var rvecQ = Utils.RotationQuatFromRodrigues(rvec);

            // Adjust the position based on image resolution
            tvec *= (resolutionTestXY.x / resolutionTestXY.y);
            tvec.x *= (centerPointTrainXY.x / centerPointTestXY.x);
            tvec.y *= (centerPointTrainXY.y / centerPointTestXY.y);

            _mainThreadWorkQueue.Enqueue(() =>
            {
                // Stationary camera, don't require camera to world transform
                if (isDesktop)
                {
                    // Update the tracked gameobject with the relevant pose
                    PoseDataGo.transform.SetPositionAndRotation(
                        tvec,
                        rvecQ);

                    Debug.Log($"tvec: {tvec}");
                    Debug.Log($"rvecQ: {rvecQ}");
                }

                // HoloLens camera, moving so need per frame camera to world transform
                else
                {
                    // Get the transform in the world coordinate system
                    // transformUnityWorld = CameraToWorldUnity * transformUnityCamera
                    var transformUnityCamera = Matrix4x4.TRS(
                        tvec,
                        rvecQ,
                        Vector3.one);

                    // Flip y component of translation
                    var transformUnityWorld = CameraToWorldUnity * transformUnityCamera;

                    Debug.Log($"CoordinateSystemInfo.CameraToWorldUnity: {CameraToWorldUnity}");
                    Debug.Log($"transformUnityCamera: {transformUnityCamera}");
                    Debug.Log($"transformUnityWorld: {transformUnityWorld}");

                    var tvecAdj = Utils.GetVectorFromMatrix(transformUnityWorld);
                    var rvecQAdj = Utils.GetQuatFromMatrix(transformUnityWorld);

                    // Update the tracked gameobject with the relevant pose
                    PoseDataGo.transform.SetPositionAndRotation(
                        tvecAdj,
                        rvecQAdj);

                    Debug.Log($"tvecAdj: {tvecAdj}");
                    Debug.Log($"rvecQAdj: {rvecQAdj}");
                }
            });
        }
    }
}
