# stenden_debugging

My debugging solutions to stenden's required MaCV&amp;DS application. Approach to each problem is written below. q1-4.py contain the separate question solutions. The jupyter also has the same. 

All .py files are formatted using [ruff](https://docs.astral.sh/ruff/).

---
# Question 1.

<font size="4px"><p>This method returns the fruit name by getting the string at a specific index of the set.</p>
<dl>
<dt>param fruit_id</dt>
<dd><p>The id of the fruit to get</p>
</dd>
<dt>param fruits</dt>
<dd><p>The set of fruits to choose the id from</p>
</dd>
<dt>return</dt>
<dd><p>The string corrosponding to the index <code>fruit_id</code></p>
</dd>
</dl>
<p><strong>This method is part of a series of debugging exercises.</strong> <strong>Each Python method of this series contains bug that needs to be found.</strong></p>
<div class="line-block"><code>1   It does not print the fruit at the correct index, why is the returned result wrong?</code><br />
<code>2   How could this be fixed?</code></div>
<p>This example demonstrates the issue: name1, name3 and name4 are expected to correspond to the strings at the indices 1, 3, and 4: 'orange', 'kiwi' and 'strawberry'..</p>
</font>

## Answer:

1. Sets are unordered so it always goes in with a different order.
<br>
2. Use lists instead of set when calling the function, e.g. ```name1 = id_to_fruit(1, ["apple", "orange", "melon", "kiwi", "strawberry"])```, and change the function prototype to match the new change ```def id_to_fruit(fruit_id: int, fruits: list[str]) -> str:```.
---

# Question 2
<font size="4px"><p>This method will flip the x and y coordinates in the coords array.</p>
<dl>
<dt>param coords</dt>
<dd><p>A numpy array of bounding box coordinates with shape [n,5] in format: :</p>
<pre><code>[[x11, y11, x12, y12, classid1],
 [x21, y21, x22, y22, classid2],
 ...
 [xn1, yn1, xn2, yn2, classid3]]</code></pre>
</dd>
<dt>return</dt>
<dd><p>The new numpy array where the x and y coordinates are flipped.</p>
</dd>
</dl>
<p><strong>This method is part of a series of debugging exercises.</strong> <strong>Each Python method of this series contains bug that needs to be found.</strong></p>
<div class="line-block"><code>1   Can you spot the obvious error?</code><br />
<code>2   After fixing the obvious error it is still wrong, how can this be fixed?</code></div>
</font>

## Answer

1. The first error is the swapping of the first two cells, i.e. `x11` &harr; `y11` , `x21` &harr; `y21` etc.
    - The fix is to change the second `coords[:, 1]` to `coords[:, 0]`
2. Even after the error, the function is still wrong, as the assignment happens sequentially, based on the same input. 
    - To simplify, `coords[:, 0] = coords[:, 1]` executes first, so `coords[:,0]` is now `[5,3,3,4,5]`.
    - Then `coords[:, 1]` = `coords[:, 0]` now reads from this updated `coords[:, 0]`, thus duplicating itself.
    - This current form of swapping is highly unreadable.
    - To fix this, a deep copy is an obvious workaround with the current list slicing method, with import copy. 
    - An alternate way would be to utilize for loops and would probably be frowned upon by some.


---

# Question 3.
<font size="4px"><p>This code plots the precision-recall curve based on data from a .csv file, where precision is on the x-axis and recall is on the y-axis. It it not so important right now what precision and recall means.</p>
<dl>
<dt>param csv_file_path</dt>
<dd><p>The CSV file containing the data to plot.</p>
</dd>
</dl>
<p><strong>This method is part of a series of debugging exercises.</strong> <strong>Each Python method of this series contains bug that needs to be found.</strong></p>
<div class="line-block"><code>1   For some reason the plot is not showing correctly, can you find out what is going wrong?</code><br />
<code>2   How could this be fixed?</code></div>
<p>This example demonstrates the issue. It first generates some data in a csv file format and the plots it using the <code>plot_data</code> method. If you manually check the coordinates and then check the plot, they do not correspond.</p>
</font>

## Answer
1. First, there is an error when reading the file, at least on windows. w.writerow() appends a new newline after each entry,
    - This causes results to have empty arrays like [[], [0.013, 0.951], ...]. This is fixed in my code by ignoring empty lists when iterating.
    - The values are read in as strings, not floats. So matplotlib has issues when ordering these plots(results.append([float(element) for element in row]))
    - The fix for this is to convert them to floats using list comprehension.

2. The second issue is that the `plt.plot(...)` call uses the x-axis for Y-axis in its plot and vice-versa. This does not make a change in the shape of the curve but the norm should be [x,y] when passing coordinates to .plot(x-points, y-points). The changes are as follows
    - It should be `plt.plot(results[:, 0], results[:, 1])` instead of `plt.plot(results[:, 1], results[:, 0])`
    - Additionally, the `xlabel` and `ylabel` should be swapped.

---
# Question 4
<font size="4px"><p>The method trains a Generative Adversarial Network and is based on: <a href="https://realpython.com/generative-adversarial-networks/">https://realpython.com/generative-adversarial-networks/</a></p>
<p>The Generator network tries to generate convincing images of handwritten digits. The Discriminator needs to detect if the image was created by the Generater or if the image is a real image from a known dataset (MNIST). If both the Generator and the Discriminator are optimized, the Generator is able to create images that are difficult to distinguish from real images. This is goal of a GAN.</p>
<p>This code produces the expected results at first attempt at about 50 epochs.</p>
<dl>
<dt>param batch_size</dt>
<dd><p>The number of images to train in one epoch.</p>
</dd>
<dt>param num_epochs</dt>
<dd><p>The number of epochs to train the gan.</p>
</dd>
<dt>param device</dt>
<dd><p>The computing device to use. If CUDA is installed and working then <span class="title-ref">cuda:0</span> is chosen otherwise 'cpu' is chosen. Note: Training a GAN on the CPU is very slow.</p>
</dd>
</dl>
<p><strong>This method is part of a series of debugging exercises.</strong> <strong>Each Python method of this series contains bug that needs to be found.</strong></p>
<p>It contains at least two bugs: one structural bug and one cosmetic bug. Both bugs are from the original tutorial.</p>
<div class="line-block"><code>1   Changing the batch_size from 32 to 64 triggers the structural bug.</code><br />
<code>2   Can you also spot the cosmetic bug?</code><br />
<code>Note: to fix this bug a thorough understanding of GANs is not necessary.</code></div>
<p>Change the batch size to 64 to trigger the bug with message: ValueError: "Using a target size (torch.Size([128, 1])) that is different to the input size (torch.Size([96, 1])) is deprecated. Please ensure they have the same size."</p>
</font>

## Answer
1. The error occurs when n == 937 at the line `loss_discriminator = loss_function(output_discriminator, all_samples_labels)`, suggesting that `loss_function = nn.BCELoss()` or `nn.BCELoss()` is triggering this error. Further debugging into variables, I find that the len of all_samples in this scenario is 96. These are my findings towards a stable solution.
    - In the previous iteration, it was 128. This causes the output_discriminator to have length 96, thus raising the value error.
    - The length of real_samples of iteration 937 is 32, instead of 64. Thus the concat `torch.cat()` call combines the two to create.
        - This can be inspected by placing a conditional breakpoint at `for n, (real_samples, mnist_labels) in enumerate(train_loader):`
            The condition being `n==935` or `n==936`.
            At `n==935` we can see the size of real_samples to be 64 upon stepping into the loop.
            At n==936 we can see the size of real_samples to be 32 upon stepping into the loop.
        - More into this, the `train_loader/train_set` has 60000 data points.
            When stack_size is set to 64, it cannot evenly distribute it.
            60000/64 = 937.5, which is where the error happens, rounded down.
        - I tried the following to try and make the program run without errors being raised.
            - I tried to slice the train_set with the list slicing but it did not work.
            - I used torch's subset to slice the training set. No errors are raised, however the images don't render correctly towards the end.
            - My other approach would be to raise an error that batch size 64 is unsupported.
            - My final approach would be to, instead of slicing/trimming the initial dataset, either double up the final 32 real_samples or drop that iteration entirely.
        - Alternatively, if it is possible to pre-set the number of data points, I could instead reinstatiate the `train_set` by adding the remainder to the previously found number of datapoints. Not sure if possible.
        - On further investigation and several attempts at running the batch size 32, the size of the real samples is not used for any of the `torch` calls. Simply setting batch_size to be the length of `real_samples` seems to work.

2. The cosmetic error is most likely the arbitrary showing of figures if the condition `n == batch_size - 1` is satisfied. 
    - I previously found that with batch size set to 64, n can range from 0 to 937.5 ( `0 to 60000 / batch_size` ).
    - From this, we can derive with batch size 32, only the 32nd batch is output and the rest 1874 are discarded. Similarly for batch size 64, 936 are discarded. 
    - This seems too arbitrary. Normally, you would want to display the 1st or 2nd result + the last result to indicate a clear path from start to end for a given epoch.
    - The fix I have used is to  display/output the image if it is the first or last iteration in the inner loop of the epoch loop.
