// Download ONNX model from https://github.com/onnx/models/blob/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx
// to project directory before run

using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using arcfacelib;

internal class Program {
    private static async Task Main(string[] args) {
        try {
            var arc = new ArcFace();
            Console.WriteLine("Predicting contents of image..."); 

            foreach(var kv in arc.InputMetadata)
                Console.WriteLine($"{kv.Key}: {MetadataToString(kv.Value)}");
            foreach(var kv in arc.OutputMetadata)
                Console.WriteLine($"{kv.Key}: {MetadataToString(kv.Value)}]");

            var imagesFolderPath = Path.GetFullPath($"{Directory.GetCurrentDirectory()}/app/images");
            //Console.WriteLine(imagesFolderPath);
            string[] filenames = Directory.GetFiles(imagesFolderPath, "*.png");
            //foreach (var str in filenames) {
            //    Console.WriteLine(str);
            //}
            var images = new List<Image<Rgb24>>();
            foreach (var str in filenames)
            {
                images.Add(Image.Load<Rgb24>(str));
            }
            var canceleationToken = new CancellationToken();
        
            var distmatrix = await arc.GetDistanceMatrix(images.ToArray(), canceleationToken);
            var simmatrix = await arc.GetSimilarityMatrix(images.ToArray(), canceleationToken);
    
            if (distmatrix != null && simmatrix != null) {
                PrintMatrix(distmatrix, "DistanceMatrix:");
                PrintMatrix(simmatrix, "SimilarityMatrix:");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine(ex.Message);
        }

        string MetadataToString(NodeMetadata metadata)
            => $"{metadata.ElementType}[{String.Join(",", metadata.Dimensions.Select(i => i.ToString()))}]";

        void PrintMatrix(float[,] matrix, string info) {
            Console.WriteLine(info);

            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    Console.Write(matrix[i, j]);
                    Console.Write(' ');
                }

                Console.WriteLine(Environment.NewLine);
            }
        }
    }
}