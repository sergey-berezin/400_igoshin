using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using arcfacelib;
using Ookii.Dialogs.Wpf;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace wpf
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public class ViewModel : INotifyPropertyChanged
    {
        private ArcFace? Arc;
        private string ImgFolder;
        private List<Image<Rgb24>> Images;
        private float[,]? Distances;
        private float[,]? Similarities;
        private string DistStr = "";
        private string SimStr = "";
        private double ProgressBar;
        private bool IsEnabled;
        public event PropertyChangedEventHandler? PropertyChanged;
        private CancellationTokenSource cancellationTokenSource;
        private CancellationToken cancellationToken;

        public ViewModel()
        {
            Arc = null;
            ImgFolder = "";
            Images = new List<Image<Rgb24>>();
            Distances = null;
            Similarities = null;
            ProgressBar = 0;
            IsEnabled = true;
            cancellationTokenSource = new CancellationTokenSource();
            cancellationToken = cancellationTokenSource.Token;
        }

        public string ImagesFolderPath 
        {
            get => ImgFolder;
            set
            {
                ImgFolder = value;
                OnPropertyChanged();
            }
        } 
        
        public void GetImages()
        {
            
            string[] filenames = Directory.GetFiles(ImagesFolderPath, "*.png");
            foreach (var str in filenames)
            {
                Images.Add(Image.Load<Rgb24>(str));
            }
        }

        public async Task Start() {
            try {
                IsStartEnabled = false;
                Arc = new ArcFace();
                GetImages();

                float[,] distances = new float[Images.Count, Images.Count];
                float[,] similarities = new float[Images.Count, Images.Count];
                double allProgress = (double) (Images.Count * Images.Count);       

                cancellationTokenSource.TryReset();            
                for (int i = 0; i < Images.Count; i++)
                {
                    for (int j = 0; j < Images.Count; j++)
                    {
                        var emb1 = await Arc.GetEmbeddings(Images[i], cancellationToken);
                        var emb2 = await Arc.GetEmbeddings(Images[j], cancellationToken);

                        distances[i, j] = Arc.Distance(emb1, emb2);
                        similarities[i, j] = Arc.Similarity(emb1, emb2);

                        CurrentProgress = (((i * Images.Count + j + 1)) / allProgress) * 100;
                    }
                }

                Distances = distances;
                Similarities = similarities;

                if (Distances != null && Similarities != null)
                {
                    DistMatrix = mtostr(Distances);
                    SimMatrix = mtostr(Similarities);
                }
            } catch (Exception ex) {
                MessageBox.Show(ex.Message);
            } finally {
                IsStartEnabled = true;
                Clear();
            }
        }

        private string mtostr(float[,] matrix)
        {
            string str = "";
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    str += matrix[i, j].ToString() + ' ';
                }

                str += '\n';
            }
            return str;
        }

        public string DistMatrix
        {
            get => DistStr;
            set 
            {
                DistStr = value;
                OnPropertyChanged();
            }
        }

        public string SimMatrix 
        {
            get => SimStr;
            set 
            {
                SimStr = value;
                OnPropertyChanged();
            }
        }

        public double CurrentProgress
        {
            get => ProgressBar;
            set 
            {
                ProgressBar = value;
                OnPropertyChanged();
            }
        }

        public bool IsStartEnabled
        {
            get => IsEnabled;
            set
            {
                IsEnabled = value;
                OnPropertyChanged();
            }
        }

        public void Cancel()
        {
            cancellationTokenSource.Cancel();
        }

        public void Clear()
        {
            Images.Clear();
            CurrentProgress = 0;
            Distances = null;
            Similarities = null;
        }

        private void OnPropertyChanged([CallerMemberName] string propertyName = "")
        {
            if (PropertyChanged != null)
                PropertyChanged(this, new PropertyChangedEventArgs(propertyName));
        }
    }

    public partial class MainWindow : Window
    {
        public ViewModel ViewModel { get; set; }
        public MainWindow()
        {
            InitializeComponent();
            ViewModel = new ViewModel();
            DataContext = ViewModel;
        }
        private void SelectButtonClick(object sender, RoutedEventArgs e)
        {
            var dialog = new VistaFolderBrowserDialog();

            if (dialog.ShowDialog() == true)
            {
                ViewModel.ImagesFolderPath = dialog.SelectedPath; 
            }
        }
        private async void StartButtonClick(object sender, RoutedEventArgs e)
        {
            await ViewModel.Start();
        }

        private void CancelButtonClick(object sender, RoutedEventArgs e)
        {
            ViewModel.Cancel();
        }
    }
}
