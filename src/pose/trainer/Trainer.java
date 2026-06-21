package pose.trainer;

import pose.encoder.Network;
import pose.layer.Tensor;

public class Trainer
{
	private Network network;
	private Dataset dataset;
	private Optimizer optimizer;
	private LossFunction lossFunction;

	public Trainer(Network network, Dataset dataset, Optimizer optimizer, LossFunction lossFunction)
	{
		this.network = network;
		this.dataset = dataset;
		this.optimizer = optimizer;
		this.lossFunction = lossFunction;
	}

	public void train(int epochs)
	{
		for (int epoch = 0; epoch < epochs; epoch++)
		{
			dataset.shuffle();
			
			float epochLoss = 0f;
			
			for(int i=0; i<dataset.size(); i++)
			{
				TrainingSet sample = dataset.get(i);
				
				Tensor prediction = network.forward(sample.getImage());
				
				float loss = lossFunction.forward(prediction, sample.getTargetHeatmaps());
				epochLoss += loss;
				
				Tensor gradient = lossFunction.backward(prediction, sample.getTargetHeatmaps());
				
				network.backward(gradient);
				optimizer.step(network);
			}
			System.out.println("Epoch " + epoch + " Loss: " + epochLoss / dataset.size());
		}
	}
}














