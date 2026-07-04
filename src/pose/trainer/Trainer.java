package pose.trainer;

import pose.layer.Layer;
import pose.layer.Tensor;

public class Trainer
{
	private Layer layer;
	private Dataset dataset;
	private Optimizer optimizer;
	private LossFunction lossFunction;

	public Trainer(Layer layer, Dataset dataset, Optimizer optimizer, LossFunction lossFunction)
	{
		this.layer = layer;
		this.dataset = dataset;
		this.optimizer = optimizer;
		this.lossFunction = lossFunction;
	}

	public void train(int epochs)
	{
		DataLoader loader = new DataLoader(dataset, 16);
		
		for(int epoch = 0; epoch < epochs; epoch++)
		{
			Batch batch = loader.next();
			
			Tensor prediction = layer.forward(batch.getImages());
			Tensor target = batch.getHeatmaps();
			
			float loss = lossFunction.forward(prediction, target);
			Tensor gradient = lossFunction.backward(prediction, target);
			
			layer.backward(gradient);
			
			optimizer.step(layer);
			
			batch.getImages().release();
			batch.getHeatmaps().release();
			prediction.release();
			target.release();
			gradient.release();
			
			System.out.println("Epoch " + epoch+1 + " | Loss: " + loss);
		}
	}
}














