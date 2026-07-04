package pose.tensor;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.Map;

public class TensorPool
{
	private final Map<String, Deque<Tensor>>pool = new HashMap<String, Deque<Tensor>>();
	
	public Tensor acquire(int batch, int channels, int height, int width)
	{
		String key = batch + "|" + channels + "|" + height + "|"+ width;
		
		Deque<Tensor>tensors = pool.get(key);
		
		if(tensors != null && !tensors.isEmpty())
		{
			Tensor tensor = tensors.pop();
			Arrays.fill(tensor.getData(), 0f);
			return tensor;
		}
		return new Tensor(batch, channels, height, width);
	}
	
	public void release(Tensor tensor)
	{
		String key = tensor.getBatchSize() + "|" + tensor.getChannels() + "|" + tensor.getHeight() + "|"+ tensor.getWidth();
		
		pool.computeIfAbsent(key, k -> new ArrayDeque<Tensor>()).push(tensor);
	}
}
