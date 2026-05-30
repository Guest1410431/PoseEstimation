package pose.layer;

import java.util.ArrayList;
import java.util.List;

public class Sequential extends Layer
{
	private final List<Layer>layers = new ArrayList<Layer>();
	
	public Sequential add(Layer layer)
	{
		layers.add(layer);
		return this;
	}
	
	@Override
	public Tensor forward(Tensor input)
	{
		Tensor tensor = input;
		
		for(Layer layer : layers)
		{
			tensor = layer.forward(tensor);
		}
		return tensor;
	}

}
