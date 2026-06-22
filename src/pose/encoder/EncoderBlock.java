package pose.encoder;

import pose.layer.ActivationLayer;
import pose.layer.ConvolutionalLayer;
import pose.layer.DownSampleLayer;
import pose.layer.Layer;
import pose.layer.Sequential;
import pose.layer.SkipConnection;
import pose.layer.Tensor;

public class EncoderBlock extends Layer
{
	private final Sequential sequential;
	private final SkipConnection skip;
	
	public EncoderBlock(int inC, int outC)
	{
		sequential = new Sequential()
				.add(new ConvolutionalLayer(inC, outC, 3, 1, 1))
				.add(new ActivationLayer())
				.add(new ConvolutionalLayer(outC, outC, 3, 1, 1))
				.add(new ActivationLayer())
				.add(new DownSampleLayer(2, 2));
		skip = new SkipConnection();
	}
	
	@Override
	public Tensor forward(Tensor input)
	{
		skip.save(input);
		return sequential.forward(input);
	}

	public SkipConnection getSkip()
	{
		return skip;
	}

	@Override
	public Tensor backward(Tensor gradient)
	{
		return sequential.backward(gradient);
	}
}
