package pose.layer;

import java.util.ArrayList;
import java.util.List;

import pose.tensor.Tensor;

public class Sequential extends Layer
{
	private final List<Layer> layers = new ArrayList<Layer>();

	public Sequential add(Layer layer)
	{
		layers.add(layer);
		return this;
	}

	@Override
	public Tensor forward(Tensor input)
	{
		Tensor tensor = input;

		for (Layer layer : layers)
		{
			tensor = layer.forward(tensor);
		}
		//System.out.println("sequential forward---min: " + tensor.min() + " | max: " + tensor.max() + " | mean: " + tensor.mean());
		return tensor;
	}

	@Override
	public Tensor backward(Tensor gradient)
	{
		for (int i = layers.size() - 1; i >= 0; i--)
		{
			gradient = layers.get(i).backward(gradient);
		}	
		//System.out.println("sequential backward---min: " + gradient.min() + " | max: " + gradient.max() + " | mean: " + gradient.mean());
		return gradient;
	}

	public void updateWeights(float learningRate)
	{
		for (Layer layer : layers)
		{
			layer.updateWeights(learningRate);
		}
	}
}
