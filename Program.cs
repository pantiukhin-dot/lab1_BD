using System;
using System.IO;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace GameSalesPrediction
{
    // Class representing the game sales dataset
    public class GameSalesData
    {
        [LoadColumn(0)]
        public int Rank { get; set; } // Ranking of overall sales

        [LoadColumn(1)]
        public string Name { get; set; } // Game's name

        [LoadColumn(2)]
        public string Platform { get; set; } // Platform of the game's release (PC, PS4, etc.)

        [LoadColumn(3)]
        public int Year { get; set; } // Year of the game's release

        [LoadColumn(4)]
        public string Genre { get; set; } // Genre of the game

        [LoadColumn(5)]
        public string Publisher { get; set; } // Publisher of the game

        [LoadColumn(6)]
        public float NA_Sales { get; set; } // Sales in North America (in millions)

        [LoadColumn(7)]
        public float EU_Sales { get; set; } // Sales in Europe (in millions)

        [LoadColumn(8)]
        public float JP_Sales { get; set; } // Sales in Japan (in millions)

        [LoadColumn(9)]
        public float Other_Sales { get; set; } // Sales in the rest of the world (in millions)

        [LoadColumn(10)]
        public float Global_Sales { get; set; } // Total worldwide sales
    }

    // Class for predicting the output of game sales
    public class GameSalesPrediction
    {
        [ColumnName("Score")]
        public float Global_Sales { get; set; }
    }

    class Program

    {
        static void Main(string[] args)
        {
            // Create a new MLContext for ML.NET operations
            var mlContext = new MLContext();

            // Load the dataset from the specified path
            string dataPath = "C:\\Users\\vgsales.csv";
            IDataView data = mlContext.Data.LoadFromTextFile<GameSalesData>(dataPath, separatorChar: ',', hasHeader: true);

            // Split the data into training and testing sets (80/20 split)
            var trainTestSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var trainingData = trainTestSplit.TrainSet; // Training data
            var testData = trainTestSplit.TestSet; // Testing data

            // Data processing pipeline
            var dataProcessPipeline = mlContext.Transforms.Conversion.ConvertType("Year")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Platform"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Genre"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Publisher"))
                .Append(mlContext.Transforms.NormalizeMinMax("NA_Sales"))
                .Append(mlContext.Transforms.NormalizeMinMax("EU_Sales"))
                .Append(mlContext.Transforms.NormalizeMinMax("JP_Sales"))
                .Append(mlContext.Transforms.NormalizeMinMax("Other_Sales"))
                .Append(mlContext.Transforms.Concatenate("Features", "Year", "Platform", "Genre", "Publisher", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"));

            // Specify the regression trainer
            var trainer = mlContext.Regression.Trainers.FastTree(
                labelColumnName: "Global_Sales",
                numberOfTrees: 100,
                minimumExampleCountPerLeaf: 5,
                numberOfLeaves: 50,
                learningRate: 0.2);

            // Create the training pipeline
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // Train the model
            var model = trainingPipeline.Fit(trainingData);

            // Evaluate the model on the test data
            var predictions = model.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Global_Sales", scoreColumnName: "Score");

            // Output model evaluation metrics
            Console.WriteLine("Model Evaluation Metrics:");
            Console.WriteLine($"Mean Absolute Error (MAE): {metrics.MeanAbsoluteError}");
            Console.WriteLine($"Root Mean Squared Error (RMSE): {metrics.RootMeanSquaredError}");
            Console.WriteLine($"R-squared (R2): {metrics.RSquared}");

            // Save the trained model to a file
            using (var fileStream = new FileStream("game_sales_model.zip", FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, trainingData.Schema, fileStream);
            }

            // Load the saved model from the file for future predictions
            ITransformer loadedModel;
            using (var fileStream = new FileStream("game_sales_model.zip", FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(fileStream, out var modelInputSchema);
            }

            // Create a prediction engine from the loaded model
            var predictionEngine = mlContext.Model.CreatePredictionEngine<GameSalesData, GameSalesPrediction>(loadedModel);

            // Sample input data for making a prediction
            var newData = new GameSalesData
            {
                Rank = 5,
                Name = "Sample Game",
                Platform = "PS4",
                Year = 2022,
                Genre = "Action",
                Publisher = "Sample Publisher",
                NA_Sales = 41.49f,
                EU_Sales = 29.02f,
                JP_Sales = 3.77f,
                Other_Sales = 8.46f
            };

            // Use the prediction engine to predict the global sales based on new data
            var prediction = predictionEngine.Predict(newData);
            Console.WriteLine($"\nPrediction for new data: Name={newData.Name}, Platform={newData.Platform}, Year={newData.Year}, Genre={newData.Genre}, Publisher={newData.Publisher}");
            Console.WriteLine($"Predicted Global Sales: {prediction.Global_Sales}");

            // Check if the model performance is acceptable based on R-squared value
            if (metrics.RSquared >= 0.7)
            {
                Console.WriteLine("The model performs well and can be used for prediction on this type of data.");
            }
            else
            {
                Console.WriteLine("The model may not be accurate enough for reliable predictions. Consider improving the model.");
            }

            Console.WriteLine("Press Enter to exit...");
            Console.ReadLine();
        }
    }
}
