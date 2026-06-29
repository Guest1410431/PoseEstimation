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
	private final Sequential featureExtractor;
	private final DownSampleLayer downSample;
	private final SkipConnection skip;
	
	public EncoderBlock(int inC, int outC)
	{
		featureExtractor = new Sequential()
		        .add(new ConvolutionalLayer(inC, outC, 3, 1, 1))
		        .add(new ActivationLayer())
		        .add(new ConvolutionalLayer(outC, outC, 3, 1, 1))
		        .add(new ActivationLayer());

		downSample = new DownSampleLayer(2, 2);
		skip = new SkipConnection();
	}
	
	@Override
	public Tensor forward(Tensor input)
	{
		Tensor features = featureExtractor.forward(input);
		skip.save(features);
		return downSample.forward(features);
	}

	public SkipConnection getSkip()
	{
		return skip;
	}

	@Override
	public Tensor backward(Tensor gradient)
	{
		Tensor pooledGradient = downSample.backward(gradient);
		Tensor skipGradient = skip.getGradient();
		
		if(skipGradient != null)
		{
			pooledGradient.add(skipGradient);
		}
		skip.clear();
		
		return featureExtractor.backward(pooledGradient);
	}
}
