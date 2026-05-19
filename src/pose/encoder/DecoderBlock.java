package pose.encoder;

import pose.layer.ActivationLayer;
import pose.layer.ConvolutionalLayer;
import pose.layer.Layer;
import pose.layer.Sequential;
import pose.layer.SkipConnection;
import pose.layer.Tensor;
import pose.layer.UpSampleLayer;

public class DecoderBlock extends Layer
{
	private final Sequential sequential;
	private final SkipConnection skip;
	
	public DecoderBlock(int inC, int outC, SkipConnection skip)
	{
		this.skip = skip;
		
		sequential = new Sequential()
				.add(new UpSampleLayer(2))
				.add(new ConvolutionalLayer(inC +outC, outC, 3, 1, 1))
				.add(new ActivationLayer());

	}
	@Override
	public Tensor forward(Tensor input)
	{
		Tensor skipTensor = skip.getEncoderOutput();
		Tensor merged = concat(input, skipTensor);
		return sequential.forward(merged);
	}
	
	private Tensor concat(Tensor input, Tensor skipTensor)
	{
		// TODO: channel-wise concatenation
		
		return input;
	}
}
