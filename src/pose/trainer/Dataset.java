package pose.trainer;

import java.util.Collections;
import java.util.List;

public class Dataset
{
	private List<TrainingSet> samples;

	public Dataset(List<TrainingSet> samples)
	{
		this.samples = samples;
	}

	public int size()
	{
		return samples.size();
	}
	public TrainingSet get(int index)
	{
		return samples.get(index);
	}
	public void shuffle()
	{
		Collections.shuffle(samples);
	}
}

