using System;
using System.IO;
using System.Collections.Generic;
using System.Diagnostics;


namespace MLP
{
    class Program
    {
        static void Main(string[] args)
        {
            int epoch = 100;
            int counter = 0;
            float totalSquare = 0;
            Node network = new Node();
            List<List<float>> squares = new List<List<float>>();
            BiasNode node3 = new BiasNode(1);
            HiddenNode node4 = new HiddenNode(0.74f, 0.8f, 0.35f, 0.9f);
            HiddenNode node5 = new HiddenNode(0.13f, 0.4f, 0.97f, 0.45f);
            HiddenNode node6 = new HiddenNode(0.68f, 0.1f, 0.96f, 0.36f);
            BiasNode node7 = new BiasNode(1);
            OutputNode node8 = new OutputNode(0.35f, 0.5f, 0.9f, 0.98f);
            OutputNode node9 = new OutputNode(0.8f, 0.13f, 0.8f, 0.92f);
            InputNode inputNode1 = new InputNode();
            InputNode inputNode2 = new InputNode();
            InputNode inputNode3 = new InputNode();

            //Train
            string[] lines = System.IO.File.ReadAllLines(@"data-CMP2020M-item1-train.txt");
            while (counter < epoch)
            {
                totalSquare = 0;
                //Console.WriteLine("Epoch: " + (counter+1));
                foreach (string line in lines)
                {
                    string[] data;
                    data = line.Split(' ');
                    inputNode1.updateInput(float.Parse(data[0]));
                    inputNode2.updateInput(float.Parse(data[1]));
                    inputNode3.updateInput(float.Parse(data[2]));
                    node8.changeOutput(Int32.Parse(data[3]));
                    node9.changeOutput(Int32.Parse(data[4]));
                    network.ClearNodes();
                    List<Node> step1 = new List<Node> { inputNode1, inputNode2, inputNode3, node3 };
                    network.AddChild(step1);
                    List<Node> step2 = new List<Node> { node4, node5, node6, node7 };
                    network.AddChild(step2);
                    List<Node> step3 = new List<Node> { node8, node9 };
                    network.AddChild(step3);


                    network.ForwardStep();
                    network.BackwardStep();
                    network.newWeights();

                    totalSquare += network.square();
                    //Console.WriteLine("");
                }

                List<float> newSet = new List<float>();
                squares.Add(newSet);
                squares[counter].Add(counter + 1);
                squares[counter].Add(totalSquare);
                counter++;
                
            }

            //Test
            lines = System.IO.File.ReadAllLines(@"data-CMP2020M-item1-test.txt");
            foreach (string line in lines)
            {
                string[] data;
                data = line.Split(' ');
                inputNode1.updateInput(float.Parse(data[0]));
                inputNode2.updateInput(float.Parse(data[1]));
                inputNode3.updateInput(float.Parse(data[2]));
                network.ClearNodes();
                List<Node> step1 = new List<Node> { inputNode1, inputNode2, inputNode3, node3 };
                network.AddChild(step1);
                List<Node> step2 = new List<Node> { node4, node5, node6, node7 };
                network.AddChild(step2);
                List<Node> step3 = new List<Node> { node8, node9 };
                network.AddChild(step3);


                network.ForwardStep();

                Console.WriteLine(network.softMax(node8.getNet(), node9.getNet()));
                Console.WriteLine(network.softMax(node9.getNet(), node8.getNet()));
            }
           

            string path = String.Format(@"{0}\data.txt", Environment.CurrentDirectory);
            File.WriteAllText(path, String.Empty);
            using (StreamWriter stream = File.AppendText("data.txt"))
            {
                foreach (List<float> set in squares)
                {
                    stream.WriteLine(set[0].ToString() + ' ' + set[1].ToString());
                }
                stream.Flush();
            }

            string pathExe = String.Format(@"{0}\Graph.exe", Environment.CurrentDirectory);
            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = pathExe;
            start.UseShellExecute = false;
            start.RedirectStandardOutput = true;
            using (Process process = Process.Start(start))
            {
                using (StreamReader reader = process.StandardOutput)
                {
                    string result = reader.ReadToEnd();
                    Console.Write(result);
                }
            }
        }
    }
}
