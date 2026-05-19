package pose.layer;

public class ConvolutionalLayer extends Layer
{
	private final int inChannels;
	private final int outchannels;
	private final int kernalSize;
	private final int stride;
	private final int padding;
	private final float[]weights;
	private final float[]bias;
	
	public ConvolutionalLayer(int inChannels, int outchannels, int kernalSize, int stride, int padding)
	{
		this.inChannels = inChannels;
		this.outchannels = outchannels;
		this.kernalSize = kernalSize;
		this.stride = stride;
		this.padding = padding;
			
		int weightCount = outchannels * inChannels * kernalSize * kernalSize;
		this.weights = new float[weightCount];
		this.bias = new float[outchannels];
		
        // TODO init weights (random He/Kaiming)
	}

	@Override
	public Tensor forward(Tensor input)
	{
		return input;
	}
}
