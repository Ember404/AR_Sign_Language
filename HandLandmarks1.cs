using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Mediapipe;
using Mediapipe.Tasks.Vision.HandLandmarker;
using Mediapipe.Unity.Sample;
using UnityEngine.UI;
using System.Threading.Tasks;
using Mediapipe.Tasks.Vision.Core;
using Mediapipe.Unity;
using Mediapipe.Unity.Experimental;
using Mediapipe.Unity.CoordinateSystem;
using System;
using PassthroughCameraSamples;
using System.IO;
using UnityEngine.Networking;
using Unity.Barracuda;
using System.Linq;
using Color = UnityEngine.Color;
using Unity.Collections;
using System.Drawing;
using System.Linq;


public class HandLandmarks : MonoBehaviour
{
    [SerializeField] private WebCamTextureManager m_webCamTextureManager;
    [SerializeField] private Text m_debugText;
    public RawImage myImage;

    [Header("Sign Model (Barracuda)")]
    [SerializeField] private NNModel signModelAsset;
    [SerializeField] private Text m_signDebugText;


    // CONFIG
    const int MAX_TRAJECTORY = 20;
    const float SWIPE_THRESHOLD = 60f;
    const int COOLDOWN_FRAMES = 50;

    private int currentLetterIndex = -1;
    private bool isDynamicGesture = false;

    // Gesture state
    Queue<Vector2> trajectory = new Queue<Vector2>(MAX_TRAJECTORY);
    int cooldown = 0;
    string currentLetter = "";
    float confidence = 0f;

    private Texture2D inputTexture;

    private HandLandmarker handLandmarker;
    private HandLandmarkerOptions options;
    private HandLandmarkerResult result;

    private Model signModel;
    private IWorker signWorker;
    private readonly string[] letterLabels = new[] {
       "D","A","B","C","E","F","G","H","I","K","L","M","N","O","P","R","S","U","W","Y"
    };

    readonly HashSet<string> dynamicBase = new HashSet<string> { "A", "C", "E", "L", "N", "O", "S" };

    private IEnumerator Start()
    {
        while (m_webCamTextureManager.WebCamTexture == null)
            yield return null;

        myImage.texture = m_webCamTextureManager.WebCamTexture;
        var webTex = m_webCamTextureManager.WebCamTexture;
        inputTexture = new Texture2D(webTex.width, webTex.height, TextureFormat.RGBA32, false);

        string filePath = Path.Combine(Application.streamingAssetsPath, "hand_landmarker.task");
        UnityWebRequest request = UnityWebRequest.Get(filePath);
        yield return request.SendWebRequest();
        if (request.result != UnityWebRequest.Result.Success)
        {
            m_debugText.text = "failed to load file";
            yield break;
        }
        var modelData = request.downloadHandler.data;

        options = new HandLandmarkerOptions(
            baseOptions: new Mediapipe.Tasks.Core.BaseOptions(
                Mediapipe.Tasks.Core.BaseOptions.Delegate.CPU,
                modelAssetBuffer: modelData
            ),
            runningMode: Mediapipe.Tasks.Vision.Core.RunningMode.IMAGE,
            numHands: 1,
            minHandDetectionConfidence: 0.5f,
            minHandPresenceConfidence: 0.5f,
            minTrackingConfidence: 0.5f
        );
        handLandmarker = HandLandmarker.CreateFromOptions(options, null);
        result = HandLandmarkerResult.Alloc(options.numHands);
        signModel = ModelLoader.Load(signModelAsset);
        signWorker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, signModel);

        DetectAndDraw();
    }

    private void Update() => DetectAndDraw();



    string DetectSwipe(IList<Vector2> points)
    {
        if (points.Count < 10) return null;

        Vector2 first = points[0];
        Vector2 last = points[points.Count - 1];

        float dx = last.x - first.x;
        float dy = last.y - first.y;

        if (Mathf.Abs(dx) > Mathf.Abs(dy))
        {
            if (Mathf.Abs(dx) > SWIPE_THRESHOLD)
                return dx > 0 ? "Swipe Right" : "Swipe Left";
        }
        else
        {
            if (Mathf.Abs(dy) > SWIPE_THRESHOLD)
                return dy > 0 ? "Swipe Down" : "Swipe Up";
        }
        return null;
    }


    float Angle2D(Vector2 v1, Vector2 v2)
    {
        float dot = Vector2.Dot(v1, v2);
        float norm = v1.magnitude * v2.magnitude;
        if (norm < 1e-6f) return 0f;
        return Mathf.Acos(Mathf.Clamp(dot / norm, -1f, 1f));
    }


    float[] ComputeHandJointAngles(
    IList<Mediapipe.Tasks.Components.Containers.NormalizedLandmark> landmarks,
    int imageW,
    int imageH)
    {
        Vector2[] pts = landmarks
            .Select(lm => new Vector2(lm.x * imageW, lm.y * imageH))
            .ToArray();

        Array.Resize(ref pts, 22);
        pts[21] = new Vector2(
            landmarks[0].x * imageW,
            landmarks[9].y * imageH
        );

        (int c, int p, int n)[] triplets =
        {
        (2,1,3),(3,2,4),
        (5,0,6),(6,5,7),(7,6,8),
        (9,0,10),(10,9,11),(11,10,12),
        (13,0,14),(14,13,15),(15,14,16),
        (17,0,18),(18,17,19),(19,18,20),
        (2,0,17),
        (0,21,9)
    };

        float[] angles = new float[16];

        for (int i = 0; i < triplets.Length; i++)
        {
            var (c, p, n) = triplets[i];
            Vector2 v1 = pts[p] - pts[c];
            Vector2 v2 = pts[n] - pts[c];
            angles[i] = Angle2D(v1, v2);
        }

        return angles;
    }


    private void DetectAndDraw()
    {
        if (handLandmarker == null || m_webCamTextureManager == null)
        {
            return;
        }
        cooldown = Mathf.Max(0, cooldown - 1);
        var webTex = m_webCamTextureManager.WebCamTexture;
        if (!webTex.isPlaying || webTex.width <= 16)
        {
            m_debugText.text = "Camera inactive";
            return;
        }

        //show last letter if on cooldown
        if (cooldown > 0)
        {
            Debug.Log(currentLetter);
            m_signDebugText.text = currentLetter;
            return;
        }

        inputTexture.SetPixels32(webTex.GetPixels32());
        inputTexture.Apply();

        if (inputTexture == null || inputTexture.width <= 16 || inputTexture.height <= 16)
        {
            m_debugText.text = "Invalid texture size";
            return;
        }

        using (var frame = new TextureFrame(inputTexture.width, inputTexture.height, TextureFormat.RGBA32))
        {
            frame.ReadTextureOnCPU(inputTexture, flipHorizontally: false, flipVertically: true);
            var mediaPipeImage = frame.BuildCPUImage();



            var imageOptions = new ImageProcessingOptions(rotationDegrees: 0);
            var result = HandLandmarkerResult.Alloc(options.numHands);


            bool success = handLandmarker.TryDetect(mediaPipeImage, imageOptions, ref result);
            if (!success || result.handLandmarks.Count == 0)
            {
                Debug.Log("nie wykryto dłoni");
                m_debugText.text = "No hands detected";
                return;
            }
            m_debugText.text = "Detected hand";

            var lmList = result.handLandmarks[0].landmarks;


            //draw hand landmarks
            foreach (Transform child in myImage.transform) Destroy(child.gameObject);

            var rt = myImage.rectTransform.rect;

            foreach (var lm in result.handLandmarks[0].landmarks)
            {
                var pos = rt.GetPoint(in lm);
                pos.z = 0;

                var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                sphere.transform.SetParent(myImage.transform, worldPositionStays: false);
                sphere.transform.localScale = Vector3.one * 5f;
                sphere.transform.localPosition = pos;

            }

            // Compute joint angles and predict sign
            float[] angles = ComputeHandJointAngles(
                lmList,
                webTex.width,
                webTex.height
            );
            m_signDebugText.text = $"Joint angles:\n{string.Join("\n", angles.Select(a => a.ToString("0.00")))}";


            string debugInfo = $"Input tensor shape: (1, 16)\n" +
                               $"Min: {angles.Min():0.000}, Max: {angles.Max():0.000}\n" +
                               $"Angles:\n[{string.Join(", ", angles.Select(f => f.ToString("0.000")))}]\n";
            Debug.Log(debugInfo);

            //(batch: 1, channels: 16)
            using var inputTensor = new Tensor(1, 16, angles);

            signWorker.Execute(inputTensor);

            Tensor outputTensor = signWorker.PeekOutput();
            float[] scores = outputTensor.ToReadOnlyArray();

            var top = scores
                .Select((val, idx) => new { Index = idx, Score = val })
                .OrderByDescending(x => x.Score)
                .Take(1)
                .ToList();

            string predictionResult = "";
            for (int i = 0; i < top.Count; i++)
            {
                predictionResult += $"{i + 1}: {letterLabels[top[i].Index]} ({top[i].Score * 100f:0.0}%), ";
                currentLetter = letterLabels[top[i].Index];
            }
            Debug.Log(predictionResult);
            m_debugText.text = scores.ToString();
            m_signDebugText.text = predictionResult;

            //trajectory tracking
            int cx = (int)(lmList[0].x * webTex.width);
            int cy = (int)(lmList[0].y * webTex.height);

            trajectory.Enqueue(new Vector2(cx, cy));

            //dynamic gesture detection only for certain letters
            if (!dynamicBase.Contains(currentLetter))
            {
                return;
            }


            if (cooldown == 0)
            {
                string swipe = DetectSwipe(trajectory.ToList());

                if (!string.IsNullOrEmpty(swipe))
                {
                    string gesture = swipe;
                    cooldown = COOLDOWN_FRAMES;
                    trajectory.Clear();

                    Console.WriteLine($"{gesture}");
                    currentLetter = currentLetter switch
                    {
                        "N" => "N'",
                        "A" => "A,",
                        "C" => "C'",
                        "E" => "E,",
                        "S" => "S'",
                        "L" => "L/",
                        "O" => "O'",
                        _ => ""  // default
                    };

                    Console.WriteLine(currentLetter);
                    m_signDebugText.text = currentLetter + " " + gesture; // display both letter and swipe gesture
                }
            }


            inputTensor.Dispose();
            frame.Dispose();
            mediaPipeImage.Dispose();
        }
    }

    private void OnDestroy()
    {
        inputTexture = null;
    }
}
