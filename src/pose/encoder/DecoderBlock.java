package pose.encoder;

import pose.layer.ActivationLayer;
import pose.layer.ConvolutionalLayer;
import pose.layer.Layer;
import pose.layer.Sequential;
import pose.layer.SkipConnection;
import pose.tensor.Tensor;
import pose.layer.UpSampleLayer;

public class DecoderBlock extends Layer
{
	private final Sequential sequential;
	private final UpSampleLayer upsample;
	private final SkipConnection skip;

	private int decoderChannels;
	private int skipChannels;

	public DecoderBlock(int inC, int outC, SkipConnection skip)
	{
		this.skip = skip;

		upsample = new UpSampleLayer(2);

		sequential = new Sequential().add(new ConvolutionalLayer(inC + outC, outC, 3, 1, 1)).add(new ActivationLayer()).add(new ConvolutionalLayer(outC, outC, 3, 1, 1)).add(new ActivationLayer());
	}

	@Override
	public Tensor forward(Tensor input)
	{
		Tensor up = upsample.forward(input);
		Tensor skipTensor = skip.getEncoderOutput();

		decoderChannels = up.getChannels();
		skipChannels = skipTensor.getChannels();

		Tensor merged = concat(up, skipTensor);
		return sequential.forward(merged);
	}

	private Tensor concat(Tensor input, Tensor skipTensor)
	{
		if (input.getBatchSize() != skipTensor.getBatchSize() || input.getHeight() != skipTensor.getHeight() || input.getWidth() != skipTensor.getWidth())
		{
			throw new IllegalArgumentException("Cannot concatenate tensors with different spatial dimensions");
		}
		int batchSize = input.getBatchSize();
		int inChannels = input.getChannels();
		int skipChannels = skipTensor.getChannels();

		int height = input.getHeight();
		int width = input.getWidth();

		int channelSize = height * width;

		Tensor output = Tensor.acquire(batchSize, inChannels + skipChannels, height, width);

		float[] in = input.getData();
		float[] skip = skipTensor.getData();
		float[] out = output.getData();

		int inputBatchSize = inChannels * channelSize;
		int skipBatchSize = skipChannels * channelSize;
		int outputBatchSize = (inChannels + skipChannels) * channelSize;

		for (int batch = 0; batch < output.getBatchSize(); batch++)
		{
			int inputBatchOffset = batch * inputBatchSize;
			int skipBatchOffset = batch * skipBatchSize;
			int outputBatchOffset = batch * outputBatchSize;
			// Copy decoder channels
			System.arraycopy(in, inputBatchOffset, out, outputBatchOffset, inputBatchSize);
			// Copy skip channels
			System.arraycopy(skip, skipBatchOffset, out, outputBatchOffset + inputBatchSize, skipBatchSize);
		}
		return output;
	}

	@Override
	public Tensor backward(Tensor gradient)
	{
		Tensor mergedGradient = sequential.backward(gradient);

		Tensor decoderGradient = Tensor.acquire(mergedGradient.getBatchSize(), decoderChannels, mergedGradient.getHeight(), mergedGradient.getWidth());
		Tensor skipGradient = Tensor.acquire(mergedGradient.getBatchSize(), skipChannels, mergedGradient.getHeight(), mergedGradient.getWidth());

		float[] merged = mergedGradient.getData();
		float[] decoder = decoderGradient.getData();
		float[] skip = skipGradient.getData();

		int channelSize = mergedGradient.getHeight() * mergedGradient.getWidth();

		int decoderBatchSize = decoderChannels * channelSize;
		int skipBatchSize = skipChannels * channelSize;
		int mergedBatchSize = (decoderChannels + skipChannels) * channelSize;

		for (int batch = 0; batch < mergedGradient.getBatchSize(); batch++)
		{
			int mergedOffset = batch * mergedBatchSize;
			int decoderOffset = batch * decoderBatchSize;
			int skipOffset = batch * skipBatchSize;

			System.arraycopy(merged, mergedOffset, decoder, decoderOffset, decoderBatchSize);
			System.arraycopy(merged, mergedOffset + decoderBatchSize, skip, skipOffset, skipBatchSize);
		}
		this.skip.setGradient(skipGradient);

		return upsample.backward(decoderGradient);
	}
}
