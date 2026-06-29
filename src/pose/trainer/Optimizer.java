package pose.trainer;

import pose.layer.Layer;

public interface Optimizer
{
	 void step(Layer layer);
}
