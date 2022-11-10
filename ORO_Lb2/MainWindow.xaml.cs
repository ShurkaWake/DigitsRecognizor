using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Dnn;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using Tensorflow.Operations.Activation;

namespace ORO_Lb2
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    /// 
    
    public partial class MainWindow : Window
    {
        Point lastPosition;
        Emgu.CV.Dnn.Net net;

        public MainWindow()
        {
            InitializeComponent();
            lastPosition = new Point();
            net = DnnInvoke.ReadNetFromONNX("mnist-12.onnx");
            //net = DnnInvoke.ReadNetFromONNX("mnist_model.onnx");
        }

        private void Canvas_MouseMove(object sender, MouseEventArgs e)
        {
            if(e.MouseDevice.LeftButton == MouseButtonState.Pressed)
            {
                var el = new System.Windows.Shapes.Ellipse
                {
                    Width = 40,
                    Height = 40,
                    Fill = Brushes.Black,
                    Margin = new Thickness(e.GetPosition(DrawCanvas).X - 20, e.GetPosition(DrawCanvas).Y - 20, 0, 0)
                };
                DrawCanvas.Children.Add(el);
                lastPosition = e.GetPosition(DrawCanvas);
            }
        }

        private void DrawCanvas_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            lastPosition = e.GetPosition(DrawCanvas);
        }

        private void ClearButton_Click(object sender, RoutedEventArgs e)
        {
            DrawCanvas.Children.Clear();
            PredictionLabel.Content = "-";
        }

        private void LoadButton_Click(object sender, RoutedEventArgs e)
        {
            var fileDialog = new Microsoft.Win32.OpenFileDialog();
            fileDialog.Filter = "All supported graphics|*.jpg;*.jpeg;*.png|" +
                                "JPEG (*.jpg;*.jpeg)|*.jpg;*.jpeg|" +
                                "Portable Network Graphic (*.png)|*.png";

            var result = fileDialog.ShowDialog();

            if (result == true)
            {
                Image image = new Image
                {
                    Source = new BitmapImage(new Uri(fileDialog.FileName)),
                    Height = 300,
                    Width = 300
                };
                DrawCanvas.Children.Clear();
                DrawCanvas.Children.Add(image);
            }
        }

        private void ProcessButton_Click(object sender, RoutedEventArgs e)
        {
            string result = "";

            if ((bool) OwnButton.IsChecked)
            {
                result = MyMl();
            }
            else if((bool) MsintButton.IsChecked)
            {
                try
                {
                    result = Mnist();
                }
                catch
                {
                    result = "err";
                }
            }
            else
            {
                try
                {
                    result = Combination();
                }
                catch
                {
                    result = "err";
                }
            }

            PredictionLabel.Content = result;
            System.Threading.Thread.Sleep(1000);
        }

        private string Combination()
        {
            RenderToPNGFile(DrawCanvas, "MyML.bmp");
            var imageBytes = File.ReadAllBytes("MyML.bmp");
            DigitModel.ModelInput sampleData = new DigitModel.ModelInput()
            {
                ImageSource = imageBytes,
            };
            var myMlResult = DigitModel.Predict(sampleData);
            
            
            var imgToProcess = GetImageFromCanvas();
            var netInput = DnnInvoke.BlobFromImage(imgToProcess);
            net.SetInput(netInput);
            var netOutput = net.Forward();
            var netOutputArray = new float[10];
            netOutput.CopyTo(netOutputArray);
            var res = SoftMax(netOutputArray);

            return Combine(myMlResult.Score, res);
        }

        private string Combine(float[] my, float[] ms)
        {
            int[] indeces = new int[] { 0, 6, 3, 8, 1, 7, 4, 5, 2, 9 };
            float[] floats = new float[10];

            for(int i = 0; i < 10; i++)
            {
                floats[i] = (ms[i] + my[indeces[i]]) / 2f;
            }

            return GetResult(floats);
        }

        private string MyMl()
        {
            RenderToPNGFile(DrawCanvas, "MyML.bmp");
            var imageBytes = File.ReadAllBytes("MyML.bmp");
            DigitModel.ModelInput sampleData = new DigitModel.ModelInput()
            {
                ImageSource = imageBytes,
            };

            var result = DigitModel.Predict(sampleData);
            return result.PredictedLabel;
        }

        private string Mnist()
        {
            var img = GetImageFromCanvas();
            var input = DnnInvoke.BlobFromImage(img);
            net.SetInput(input);
            var output = net.Forward();
            var array = new float[10];
            output.CopyTo(array);
            var res = SoftMax(array);

            return GetResult(res);
        }

        private float[] SoftMax(float[] arr)
        {
            var exp = (from a in arr
                       select (float)Math.Exp(a))
                      .ToArray();
            var sum = exp.Sum();

            return exp.Select(x => x / sum).ToArray();

        }

        private string GetResult(float[] arr)
        {
            float max = float.MinValue;
            int index = -1;

            for(int i = 0; i < arr.Length; i++)
            {
                if (arr[i] > max)
                {
                    max = arr[i];
                    index = i;
                }
            }

            return index.ToString();
        }

        private Image<Gray, byte> GetImageFromCanvas()
        {
            RenderToPNGFile(DrawCanvas, "MnistRaw.bmp");
            System.Drawing.Bitmap bmp = new System.Drawing.Bitmap("MnistRaw.bmp");
            var temp = bmp.ToImage<Gray, byte>()
                .Not()
                .SmoothGaussian(3)
                .Resize(28, 28, Emgu.CV.CvEnum.Inter.Cubic);
            temp.Save("MnistNonBinary.bmp");

            System.Drawing.Bitmap bitmap = new System.Drawing.Bitmap("MnistNonBinary.bmp");
            System.Drawing.Bitmap res = new System.Drawing.Bitmap(bitmap.Width, bitmap.Height);
            for (int i = 0; i < bitmap.Width; i++)
            {
                for (int j = 0; j < bitmap.Height; j++)
                {
                    var pixel = bitmap.GetPixel(i, j);
                    if ((pixel.R + pixel.G + pixel.B) / 3.0 < 128)
                    {
                        res.SetPixel(i, j, System.Drawing.Color.Black);
                    }
                    else
                    {
                        res.SetPixel(i, j, System.Drawing.Color.White);
                    }
                }
            }
            return res.ToImage<Gray, byte>().Mul(1 / 255.0);
        }

        public void RenderToPNGFile(Visual targetControl, string filename)
        {
            var renderTargetBitmap = GetRenderTargetBitmapFromControl(targetControl);

            var encoder = new PngBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(renderTargetBitmap));

            var result = new BitmapImage();

            try
            {
                using (var fileStream = new FileStream(filename, FileMode.Create))
                {
                    encoder.Save(fileStream);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"There was an error saving the file: {ex.Message}");
            }
        }

        private BitmapSource GetRenderTargetBitmapFromControl(Visual targetControl, double dpi = 96.0)
        {
            if (targetControl == null) return null;

            var bounds = VisualTreeHelper.GetDescendantBounds(targetControl);
            var renderTargetBitmap = new RenderTargetBitmap((int)(bounds.Width * dpi / 96.0),
                                                            (int)(bounds.Height * dpi / 96.0),
                                                            dpi,
                                                            dpi,
                                                            PixelFormats.Pbgra32);

            var drawingVisual = new DrawingVisual();

            using (var drawingContext = drawingVisual.RenderOpen())
            {
                var visualBrush = new VisualBrush(targetControl);
                drawingContext.DrawRectangle(visualBrush, null, new Rect(new Point(), bounds.Size));
            }

            renderTargetBitmap.Render(drawingVisual);
            return renderTargetBitmap;
        }

        public System.Drawing.Bitmap ResizeImage(System.Drawing.Image image, int width, int height)
        {
            var destRect = new System.Drawing.Rectangle(0, 0, width, height);
            var destImage = new System.Drawing.Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = System.Drawing.Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, System.Drawing.GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }
    }
}
