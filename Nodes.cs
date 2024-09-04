using System;
using System.Collections.Generic;
using System.Text;

namespace MLP
{
    class Node
    {
        private List<List<Node>> steps = new List<List<Node>>();
        protected float signalOf;
        protected float net = 0;
        protected float sigmoidOutput;
        protected float error = 0;
        protected float errorSum = 0;
        protected float weightChange = 0;
        protected float expOutput;
        protected float learningRate = 0.1f;
        protected float squareError = 0;
        protected float totalSquare = 0;
        protected float softMaxValue = 0;

        protected List<float> weights = new List<float>();
        public void AddChild(List<Node> step)
        {
            steps.Add(step);
        }
        public void ClearNodes()
        {
            steps.Clear();
        }
        public int size() { return steps.Count; }

        public void ForwardStep()
        {
            for (int i = 0; i < steps.Count; i++)
            {
                if(i != 0)
                {
                    for (int j = 0; j < steps[i].Count; j++)
                    {
                        steps[i][j].net = 0;
                        for (int k = 0; k < steps[i - 1].Count; k++)
                        {
                            steps[i][j].netCalc(steps[i - 1][k].signalOf, k);
                        } 
                        steps[i][j].outputCalc();
                    }
                }
            }
        }

        public void BackwardStep()
        {
            for(int i = steps.Count - 1; i > -1; i--)
            {
                for (int j = 0; j < steps[i].Count; j++)
                {
                    if(i != steps.Count - 1 && i != 0)
                    {
                        steps[i][j].errorSum = 0;
                        for (int k = 0; k < steps[i + 1].Count; k++)
                        {
                            steps[i][j].errorSumCalc(steps[i + 1][k].error, steps[i + 1][k].weights[j]);
                        }
                        steps[i][j].errorCalc();
                    }
                    else if (i == steps.Count - 1)
                    {
                        steps[i][j].errorCalc();
                    }
                    
                }
            }
        }

        public void newWeights()
        {
            for (int i = 0; i < steps.Count; i++)
            {
                if (i != 0)
                {
                    for (int j = 0; j < steps[i].Count; j++)
                    {
                        for (int k = 0; k < steps[i - 1].Count; k++)
                        {
                            steps[i][j].weightChangeCalc(steps[i - 1][k].signalOf, k);
                        }
                    }
                }
            }
        }

        public float softMax(float net1, float net2)
        {

            softMaxValue = (float)(Math.Exp(net1) / (Math.Exp(net1) + Math.Exp(net2)));
            return softMaxValue;
        }

        public float square()
        {
            squareError = 0;
            for (int i = 0; i < steps[steps.Count-1].Count; i++)
            {
                squareError += steps[steps.Count-1][i].squareErrorCalc();
            }
            totalSquare = squareError * 0.5f;
            //Console.WriteLine(totalSquare);
            return totalSquare;
        }

        protected virtual void netCalc(float signalTo, int i){ }
        protected virtual void outputCalc(){ }
        protected virtual void errorSumCalc(float previousError, float previousWeight) { }
        protected virtual void errorCalc() { }
        protected virtual void weightChangeCalc(float signalTo, int i) { }
        protected virtual float squareErrorCalc() { return 0; }


    }

    class InputNode : Node
    {
        public float getSignal() { return signalOf; }

        public void updateInput(float input)
        {
            signalOf = input;
        }
    }

    class HiddenNode : Node
    {
        public HiddenNode(params float[] weight)
        {
            signalOf = 0;
            for (int i = 0; i < weight.Length; i++)
            {
                weights.Add(weight[i]);
            }
        }

        protected override void netCalc(float signalTo, int i)
        {
            net = net + (signalTo * weights[i]);
        }

        protected override void outputCalc()
        {
            sigmoidOutput = (float)(1 / (1 + Math.Exp(-net)));
            signalOf = sigmoidOutput;
            //Console.WriteLine(sigmoidOutput);
        }
        protected override void errorSumCalc(float previousError, float previousWeight)
        {
            errorSum += previousError * previousWeight;
        }
        protected override void errorCalc()
        {
            error = signalOf * (1 - signalOf) * errorSum;
        }
        protected override void weightChangeCalc(float signalTo, int i)
        {
            weightChange = learningRate * error * signalTo;
            //Console.Write(weightChange + "+" + weights[i] + "=");
            weights[i] = weights[i] + weightChange;
            //Console.WriteLine(weights[i]);
        }
        public float getSignal() { return signalOf; }
        public float getNet() { return net; }

        public float getWeight() { return weights[0]; }
    }

    class OutputNode : Node
    {
        public OutputNode(params float[] weight)
        {
            signalOf = 0;
            for (int i = 0; i < weight.Length; i++)
            {
                weights.Add(weight[i]);
            }
        }

        protected override void netCalc(float signalTo, int i)
        {
            net = net + (signalTo * weights[i]);
            //Console.WriteLine(net);
        }

        protected override void errorCalc()
        {
            error = expOutput - net;
            //Console.WriteLine(error);
        }
        protected override void weightChangeCalc(float signalTo, int i)
        {
            weightChange = learningRate * error * signalTo;
            //Console.Write(weightChange + "+" + weights[i] + "=");
            weights[i] = weights[i] + weightChange;
            //Console.WriteLine(weights[i]);
        }

        protected override float squareErrorCalc()
        {
            return (expOutput - net)*(expOutput - net);
        }
        public void changeOutput(int output)
        {
            expOutput = output;
        }

        public float getNet()
        {
            return net;
        }
    }

    class BiasNode : Node
    {
        public BiasNode(float signal)
        {
            error = 0;
            signalOf = signal;
        }
    
    }
}
