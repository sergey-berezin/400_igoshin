using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace arcfacelib {
    public class ArcFace : IDisposable {
        private const string ModelPath = "arcfacelib/arcfaceresnet100-8.onnx";
        private InferenceSession Session;
        public ArcFace() {
            Session = new InferenceSession(ModelPath);
            ArgumentNullException.ThrowIfNull(Session);
        }

        public void Dispose() => Session.Dispose();

        private static void CheckToken(CancellationToken token)
        {
            if (token.IsCancellationRequested)
            {
                token.ThrowIfCancellationRequested();
            }
        }

        private DenseTensor<float> ImageToTensor(Image<Rgb24> img)
        {
            var w = img.Width;
            var h = img.Height;
            var t = new DenseTensor<float>(new[] { 1, 3, h, w });

            img.ProcessPixelRows(pa => 
            {
                for (int y = 0; y < h; y++)
                {           
                    Span<Rgb24> pixelSpan = pa.GetRowSpan(y);
                    for (int x = 0; x < w; x++)
                    {
                        t[0, 0, y, x] = pixelSpan[x].R;
                        t[0, 1, y, x] = pixelSpan[x].G;
                        t[0, 2, y, x] = pixelSpan[x].B;
                    }
                }
            }); 
    
            return t;
        }

        public async Task<float[]> GetEmbeddings(Image<Rgb24> face, CancellationToken token) 
        {
            return await Task<float[]>.Factory.StartNew(() =>
            {
                CheckToken(token);

                var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("data", ImageToTensor(face)) };

                CheckToken(token);

                lock(Session) {
                    using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = Session.Run(inputs);
                    return Normalize(results.First(v => v.Name == "fc1").AsEnumerable<float>().ToArray());
                }
            } , token, TaskCreationOptions.LongRunning, TaskScheduler.Default);
        }

        public async Task<float[,]?> GetDistanceMatrix(Image<Rgb24>[] imgs, CancellationToken token) 
        {
            float[,] distanceMatrix = new float[imgs.Length, imgs.Length];
            
            try {
                CheckToken(token);
                var tasks = new List<Task<float[]>>();
                Array.ForEach(imgs, image => tasks.Add(GetEmbeddings(image, token)));
                var embeddings = await Task.WhenAll(tasks);
                
                int i = 0, j;
                foreach (var emb1 in embeddings) {
                    j = 0;
                    foreach (var emb2 in embeddings) {
                        distanceMatrix[i, j] = Distance(emb1, emb2);
                        j++;
                    }
                    i++;
                }

                return distanceMatrix;
            } catch {
                return null;
            }
        }

        public async Task<float[,]?> GetSimilarityMatrix(Image<Rgb24>[] imgs, CancellationToken token)
        {
            float[,] similarityMatrix = new float[imgs.Length, imgs.Length];
            
            try {
                CheckToken(token);
                var tasks = new List<Task<float[]>>();
                Array.ForEach(imgs, image => tasks.Add(GetEmbeddings(image, token)));
                var embeddings = await Task.WhenAll(tasks);

                int i = 0, j;
                foreach (var emb1 in embeddings) {
                    j = 0;
                    foreach (var emb2 in embeddings) {
                        similarityMatrix[i, j] = Similarity(emb1, emb2);
                        j++;
                    }
                    i++;
                }
                
                return similarityMatrix;
            } catch {
                return null;
            }
        }

        private float[] Normalize(float[] v) 
        {
            var len = Length(v);
            return v.Select(x => x / len).ToArray();
        }

        private float Length(float[] v) => (float)Math.Sqrt(v.Select(x => x*x).Sum());

        public float Distance(float[] v1, float[] v2) => Length(v1.Zip(v2).Select(p => p.First - p.Second).ToArray());

        public float Similarity(float[] v1, float[] v2) => v1.Zip(v2).Select(p => p.First * p.Second).Sum();
    
        public IReadOnlyDictionary<string, NodeMetadata> InputMetadata 
        {
            get 
            {
                return Session.InputMetadata;
            }
        }

        public IReadOnlyDictionary<string, NodeMetadata> OutputMetadata 
        {
            get 
            {
                return Session.OutputMetadata;
            }
        }
    }
}
