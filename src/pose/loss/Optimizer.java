package pose.loss;

import pose.layer.Layer;

public interface Optimizer
{
	 void step(Layer layer);
}
