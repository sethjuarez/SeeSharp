using System;
using System.Linq;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using Microsoft.ML;
using System.Collections;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.Transforms.Onnx;

namespace SeeSharp
{
    public class ImagePrediction
    {
        [ColumnName("classLabel")]
        [VectorType]
        public string[] Prediction;

        [ColumnName("loss")]
        [OnnxSequenceType]
        public List<float> Loss;
    }

    public class ImageInput
    {
        [ImageType(224, 224)]
        public Bitmap Image { get; set; }
    }


    class Program
    {
        static void Main(string[] args)
        {
            var modelFile = Directory.EnumerateFiles(".", "*.onnx").FirstOrDefault();
            Console.WriteLine($"Attempting to load {modelFile}...");
            var predictor = LoadModel(modelFile);


            Console.WriteLine("Reading Folder...");
            foreach (var file in Directory.EnumerateFiles("images", "*.jpg"))
            {
                Console.WriteLine(file);
                var output = predictor.Predict(new ImageInput { Image = (Bitmap)Image.FromFile(file) });
                Console.WriteLine($"Label: {output.Prediction}\nLoss: {output.Loss}\n\n");
            }
            
            Console.ReadKey();
        }

        public static PredictionEngine<ImageInput, ImagePrediction> LoadModel(string onnxModelFilePath)
        {
            var ctx = new MLContext();
            var dataView = ctx.Data.LoadFromEnumerable(new List<ImageInput>());

            var pipeline = ctx.Transforms.ResizeImages(
                                resizing: ImageResizingEstimator.ResizingKind.Fill, 
                                outputColumnName: "data", 
                                imageWidth: 224, 
                                imageHeight: 224, 
                                inputColumnName: nameof(ImageInput.Image))
                            .Append(ctx.Transforms.ExtractPixels(outputColumnName: "data"))
                            .Append(ctx.Transforms.ApplyOnnxModel(
                                modelFile: onnxModelFilePath, 
                                outputColumnNames: new[] { "classLabel", "loss" }, inputColumnNames: new[] { "data" }));

            var model = pipeline.Fit(dataView);
            return ctx.Model.CreatePredictionEngine<ImageInput, ImagePrediction>(model);
        }
    }
}
