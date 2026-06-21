package pose.encoder;

import java.util.List;

import pose.layer.Layer;
import pose.layer.Tensor;

public class Network
{
	private List<Layer>layers;
	
	public void add(Layer layer)
	{
		layers.add(layer);
	}
	
	public Tensor forward(Tensor input)
	{
		Tensor current = input;
		
		for(Layer layer : layers)
		{
			current = layer.forward(current);
		}
		return current;
	}
	
	public Tensor backward(Tensor gradient)
	{
		Tensor current = gradient;
		
		for(int i=layers.size()-1; i>=0; i--)
		{
			current = layers.get(i).backward(current);
		}
		return current;
	}
	
	public List<Layer> getLayers()
    {
        return layers;
    }
}
