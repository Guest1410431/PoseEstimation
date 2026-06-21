package pose.trainer;

import pose.encoder.Network;

public interface Optimizer
{
	 void step(Network network);
}
