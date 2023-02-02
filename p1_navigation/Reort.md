#### Learning Algorithm

we choice the [Rainbow](https://arxiv.org/pdf/1710.02298.pdf) method to implement this task.

the net work contains 6 ways to improve method of Dqn

​		1.[Double DQN](https://github.com/JiaRuiming-1/RL-rainbow/blob/all-contributors/add-AFanaei/02.double_q.ipynb)

​		2.[Prioritized Experience Replay (PER)](https://github.com/JiaRuiming-1/RL-rainbow/blob/all-contributors/add-AFanaei/03.per.ipynb)

​        3.[Dueling Network](https://github.com/JiaRuiming-1/RL-rainbow/blob/all-contributors/add-AFanaei/04.dueling.ipynb)

​		4.[Noisy Networks](https://github.com/JiaRuiming-1/RL-rainbow/blob/all-contributors/add-AFanaei/05.noisy_net.ipynb)

​		5.[N-Step Learning](https://github.com/JiaRuiming-1/RL-rainbow/blob/all-contributors/add-AFanaei/07.n_step_learning.ipynb)

​		6.[Categorical DQN](https://github.com/JiaRuiming-1/RL-rainbow/blob/all-contributors/add-AFanaei/06.categorical_dqn.ipynb)

#### There are many hyperparameters, the main content below:

​	train frames steps number is 60000, so that the iterations number is 200
​	memory_size (int)=1e5 length of memory
​    batch_size (int)=128 batch size for sampling
​    target_update (int)=32 period for target model's hard update
​    lr (float)=0.0001 learning rate
​    gamma (float)=0.99 discount factor
​    alpha (float)=0.5 determines how much prioritization is used
​    beta (float)=0.4 determines how much importance sampling is used
​    prior_eps (float)=1e-6 guarantees every transition can be sampled
​	v_min (float)=-10 min value of support
​    v_max (float)=10 max value of support
​    atom_size (int)=51 the unit number of support
​    n_step (int)=3 step number to calculate n-step td error

#### There are two model save in model.py

​	NoisyLinear noisy_std=0.2 the noisy network std num
​    Network: 
​	common layer is (state,256)+(256,256)
​	advantage layer is (256,128)+(128,out*atom_size)
​	value layer is (256,128)+(128,atom_size)

##### Note that Prioritized replay buffer use segment tree to choice experience samples

#### the reward plot below

![image-20230203002927797]([../../../Library/Application Support/typora-user-images/image-20230203002927797.png](https://github.com/JiaRuiming-1/deep-reinforcement-learning/blob/master/p1_navigation/%E4%B8%8B%E8%BD%BD.png))

the model saved as checkpoint_fast.pth file which trained by GPU

#### Ideas for Future Work

​	Because of the limit of state input, the agent always don't know there is a reward far from itself and trembling in place, so I want to try to use pixel state and CNN network method to train agent.

​	I need to read more papers about training gent to get some idea to try.

