# Learning meta-learning

Learning meta-learning in machine learning or learning learning how to learn.

![A chameleon changing colors 🕶](meta-learning-chameleon.gif)

_(Inspiration: meta-learning talks by Oriol Vinyals in
[2021](https://www.youtube.com/watch?v=9j4iH9TPTd8),
[2017](http://metalearning.ml/2017/))_

### Table of contents

* [Overview](#overview) * [Videos](#videos) * [Papers](#papers)

---

## Overview

Meta-learning can be described as the process of: 1) learning how to learn
([Harlow, 1949](https://psycnet.apa.org/record/1949-03097-001)); 2) a system
that discovers or improves a learning algorithm ([Hochreiter et al.,
2001](https://www.researchgate.net/publication/225182080_Learning_To_Learn_Using_Gradient_Descent));
or 3) understanding and adaptation of learning, while adjusting it according to
the task requirements ([Lemke et al.,
2013](https://link.springer.com/article/10.1007/s10462-013-9406-y)). In machine
learning, it is the study of how to use machine learning to design machine
learning methods ([Li, 2020](https://www.youtube.com/watch?v=xIwDqf8oN68)).

Traditional gradient-based ([Rumelhart et al.,
1986](http://www.cs.toronto.edu/~hinton/absps/naturebp.pdf)) deep learning
models ([Hinton, Osindero & Teh,
2006](http://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)) typically use a lot
of data for input-output measurements - black-box modelling strategies - for
specific tasks ([Hinton et al.,
2012](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/HintonDengYuEtAl-SPM2012.pdf)).
However, they: 1) tend not to perform well in many applications with limited
amounts of data and/or when they have to generalize to new tasks quickly; and 2)
require inefficient relearning of parameters to incorporate new information.

Unlike machine learning systems, humans are good at continuous ([Mitchell &
Tulukdar,
2019](https://icml.cc/media/Slides/icml/2019/hallb(10-09-15)-10-09-15-4337-never-ending_le.pdf))
learning generalizations based on one or a few examples quickly and adapt to new
circumstances, building on previous experience. Researchers have long been
working on computation models that have human-like learning capacities
([Valiant, 1984](http://web.mit.edu/6.435/www/Valiant84.pdf); [Schmidhuber,
1987](http://people.idsia.ch/~juergen/diploma1987ocr.pdf); [Y. Bengio, S. Bengio
& Cloutier, 1990](https://ieeexplore.ieee.org/document/155621/); [Thrun,
1995](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1995_1/thrun_sebastian_1995_1.pdf))
to solve tasks from a few examples and/or to make concept generalizations
similar to the way humans do. Meta-learning in machine learning focuses on
algorithm learning in a supervised learning ([Schmidhuber,
1993](https://link.springer.com/chapter/10.1007/978-1-4471-2063-6_107);
[Hochreiter et al.,
2001](https://www.researchgate.net/publication/225182080_Learning_To_Learn_Using_Gradient_Descent))
and reinforcement learning ([Schmidhuber et al.,
1996](https://www.researchgate.net/publication/2293746_Simple_Principles_Of_Metalearning))
frameworks by training meta-parameters to perform inference from limited
quantities of data in few-shot ([Finn et al.,
2017](https://arxiv.org/abs/1703.03400); [Ravi & Larochelle,
2017](https://openreview.net/pdf?id=rJY0-Kcll)) and one-shot ([Santoro,
2016](http://proceedings.mlr.press/v48/santoro16.pdf)) learning settings, in
addition to feature and model learning present in deep learning ([LeCun, Y.
Bengio & Hinton,
2015](https://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf)).

Existing meta-learning methods can help deep learning models: 1) rapidly learn
from datasets with limited data; and/or 2) learn from one task and generalize to
unseen similar tasks.

## Videos

[Oriol Vinyals (DeepMind): Perspective and Frontiers of
Meta-Learning](https://www.youtube.com/watch?v=9j4iH9TPTd8) (February 2021, AAAI
2021 Workshop on Meta-Learning)

-   [metalearning.chalearn.org](https://metalearning.chalearn.org/):
    "Meta-Learning has gained significant interest from the scientific
    community, with an increasing set of tools towards rapid learning,
    adaptation, few-shot learning, and other areas. In this talk, I'll give my
    perspective on why Meta-Learning may play a role towards natural
    intelligence and briefly describe related tasks, techniques, advances, and
    the challenges that remain."

---

[Marta Garnelo (DeepMind): Meta-Learning and Neural
Processes](https://www.youtube.com/watch?v=q-4lo5luKgc) (January 2021)

-   Abstract: "Deep neural networks excel at function approximation, yet they
    are typically trained from scratch for each new function. On the other hand,
    Bayesian methods, such as Gaussian Processes (GPs), exploit prior knowledge
    to quickly infer the shape of a new function at test time. Yet GPs are
    computationally expensive, and it can be hard to design appropriate priors.

    "We propose a family of neural models, Conditional Neural Processes (CNPs),
    that combine the benefits of both. CNPs are inspired by the flexibility of
    stochastic processes such as GPs, but are structured as neural networks and
    trained via gradient descent. CNPs make accurate predictions after observing
    only a handful of training data points, yet scale to complex functions and
    large datasets. In this talk we will introduce CNPs and their latent
    variable version ‘Neural Processes’ through the lens of meta-learning and
    discuss how they relate to a variety of existing models from this ML area."

---

[Chelsea Finn (Stanford/Google): Meta-Learning for Robustness to the Changing
World](https://www.youtube.com/watch?v=TOxvvMlxLro) (February 2021, AAAI 2021
Workshop on Meta-Learning)

-   [metalearning.chalearn.org](https://metalearning.chalearn.org/): "Machine
    learning systems are often designed under the assumption that they will be
    deployed as a static model in a single static region of the world. However,
    the world is constantly changing, such that the future no longer looks
    exactly like the past, and even in relatively static settings, the system
    may be deployed in new, unseen parts of its world. While such continuous
    shifts in the data distribution can place major challenges on models
    acquired in machine learning, the model need not be static either: it can
    and should adapt.

    "In this talk, I’ll discuss how we can allow deep networks to be robust to
    such distribution shift via adaptation. I will focus on meta-learning
    algorithms that enable this adaptation to be fast, first introducing the
    concept of meta-learning, then briefly overviewing several successful
    applications of meta-learning ranging from robotics to drug design, and
    finally discussing several recent works at the frontier of meta-learning
    research."

---

[Chelsea Finn: Multi-Task and Meta-Learning (Stanford
CS330)](https://www.youtube.com/watch?v=0rZtSwNOTQo&list=PLoROMvodv4rMC6zfYmnD7UG3LVvwaITY5)
(2019)

-   [cs330.stanford.edu](http://cs330.stanford.edu/): "While deep learning has
    achieved remarkable success in supervised and reinforcement learning
    problems, such as image classification, speech recognition, and game
    playing, these models are, to a large degree, specialized for the single
    task they are trained for. This course will cover the setting where there
    are multiple tasks to be solved, and study how the structure arising from
    multiple tasks can be leveraged to learn more efficiently or effectively.
    This includes:

    - "goal-conditioned reinforcement learning techniques that leverage the
      structure of the provided goal space to learn many tasks significantly
      faster
    - "meta-learning methods that aim to learn efficient learning algorithms
      that can learn new tasks quickly
    - "curriculum and lifelong learning, where the problem requires learning a
      sequence of tasks, leveraging their shared structure to enable knowledge
      transfer"

---

[Yoshua Bengio: Meta-learning (part of From System 1 Deep Learning to System 2
Deep Learning) (NeurIPS 2019)](https://www.youtube.com/watch?v=NgTP8DV7_zs)
(December 2019)

-   Learning in nature and meta-learning: normal learning (individual learning)
    is similar to the inner loop (training the slower time-scale
    meta-parameters, so that the model can generalize well in a new
    environment), while evolution or fast adaptation to a new environment is the
    equivalent of the outer loop (which optimises what the inner loop is
    producing).

---

[Hugo Larochelle (Google): Few-shot Learning with Meta-Learning: Progress Made
and Challenges Ahead (Machine Learning Center at Georgia
Tech)](https://www.youtube.com/watch?v=k_-IsfMiOXQ) (October 2018)

-   "A lot of the recent progress on many AI tasks enabled in part by the
    availability of large quantities of labeled data. Yet, humans are able to
    learn concepts from as little as a handful of examples. Meta-learning is a
    very promising framework for addressing the problem of generalizing from
    small amounts of data, known as few-shot learning. In meta-learning, our
    model is itself a learning algorithm: it takes input as a training set and
    outputs a classifier. For few-shot learning, it is (meta-)trained directly
    to produce classifiers with good generalization performance for problems
    with very little labeled data. In this talk, I'll present an overview of
    the recent research that has made exciting progress on this topic
    (including my own) and will discuss the challenges as well as research
    opportunities that remain."

---

## Papers

H. F. Harlow (1949). [The Formation of Learning
Sets](https://psycnet.apa.org/record/1949-03097-001). *Psychological Review,
56(1):51*.

-   "...trial and error learning theory and insight learning theory are merely
    two phases of a learning model, an initial phase and an ending phase...
    Learning sets describe the mechanisms by which complex learning problems are
    mastered by primate animals. After a time these problems are solved
    immediately or almost immediately." - H. F. Harlow
    ([1980](http://garfield.library.upenn.edu/classics1980/A1980JC93300001.pdf)).

---

S. Hochreiter, A. S. Younger, P. R. Conwell (2001). [Learning to Learn Using
Gradient
Descent](https://www.researchgate.net/publication/225182080_Learning_To_Learn_Using_Gradient_Descent).
*ICANN 2001*.

-   Demonstrates how gradient-based LSTM networks ([Hochreiter & Schmidhuber,
    1997](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735)),
    trained to meta-learn, can rapidly learn new quadratic functions with little
    data.

-   In this approach, "the learning algorithm is encoded in the weights of a
    recurrent network, but gradient descent is not performed at test time"
    ([OpenAI (Nichol et al.), 2019](https://arxiv.org/abs/1803.02999)).

---

C. Lemke, M. Budka, B. Gabrys (2013). [Metalearning: A Survey of Trends and
Technologies](https://link.springer.com/article/10.1007/s10462-013-9406-y).
*Artificial Intelligence Review, Vol. 44, pp. 117--130, 2015*.

-   Provides an overview of metalearning (or meta-learning) ([Y. Bengio, S.
    Bengio & Cloutier,
    1990](https://www.researchgate.net/publication/2383035_Learning_a_Synaptic_Learning_Rule))
    research directions as of 2013.

---

L. G. Valiant (1984). [A Theory of the
Learnable](http://web.mit.edu/6.435/www/Valiant84.pdf). *ACM, Vol. 27, Number
11, p. 1134.*

-   Demonstrates that it is possible to design learning machines which can learn
    entire (characterised) classes of concepts, which are "appropriate and
    nontrivial for general-purpose knowledge", while the "computational process
    by which the machines deduce the desired programs requires a feasible (i.e.,
    polynomial) number of steps."

-   The learner learns via accessing information in the form of "routine
    examples" and a "routine oracle" ("a human expert, a database of past
    observations, some deduction system, or a combination of these").

---

J. Schmidhuber (1987). [Evolutionary Principles in Self-Referential
Learning](http://people.idsia.ch/~juergen/diploma1987ocr.pdf). *Diploma thesis*.

-   Presents two approaches for learning how to learn (meta-learning), the
    second of which is done "on neuronal nets, associative networks, genetic
    algorithms and other 'weak' methods".

-   The thesis has "inspiring character rather than presenting a practical
    guidance to universal learning capabilities".

---

J. Schmidhuber (1993). [A 'Self-Referential' Weight
Matrix](https://link.springer.com/chapter/10.1007/978-1-4471-2063-6_107). *ICANN
1993, pp 446-450.*

-   Demonstrates a gradient-based recurrent neural network ([S. El Hihi & Y.
    Bengio,
    1992](https://papers.nips.cc/paper/1102-hierarchical-recurrent-neural-networks-for-long-term-dependencies.pdf))
    with meta-learning can make it possible for neural network weight updates
    (which are traditionally carried out by hard-wired algorithms) to be done
    with a gradient-based sequence learning algorithm, as shown in a
    (computationally-complex) recurrent network example.

---

J. Schmidhuber, J. Zhao, M. Wiering (1996). [Simple Principles of
Metalearning](https://www.researchgate.net/publication/2293746_Simple_Principles_Of_Metalearning).
*Technical Report IDSIA-69-96*.

-   Demonstrates meta-learning in a reinforcement learning context, treating
    learning algorithms like actions with action probabilities depending on the
    learner's state and policy

    For an overview of reinforcement learning, refer to [Sutton & Barto,
    1998](https://web.archive.org/web/20050806080008/http://www.cs.ualberta.ca/~sutton/book/the-book.html)).

---

C. Finn, P. Abbeel, S. Levine (2017). [Model-agnostic meta-learning for fast
adaptation of deep networks](https://arxiv.org/abs/1703.03400). *ICML 2017*.

-   Introduces the MAML (Meta-Agnostic Meta-Learning) algorithm, which learns a
    model's initial parameters - meta-initialization - to aid with faster new
    task adaptation.

-   Demonstrates that gradient-based algorithms for meta-learning allow systems
    to be more adaptable to new tasks, similar to humans.

-   (Note that, unlike in conventional deep learning, meta-learning usually
    offers to improve the algorithms not from many data instances but over
    multiple learning episodes.)

---

S. Ravi, H. Larochelle (2017). [Optimization as a Model for Few-Shot
Learning](https://openreview.net/pdf?id=rJY0-Kcll). *ICLR 2017*.

-   Proposes an LSTM ([Hochreiter & Schmidhuber,
    1997](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735))
    RNN-based ([S. El Hihi & Y. Bengio,
    1992](https://papers.nips.cc/paper/1102-hierarchical-recurrent-neural-networks-for-long-term-dependencies.pdf))
    meta-learned optimizer used to train another classifier in a few-short
    learning setting.

-   The performance is limited when fine-tuning with a few examples ([Finn,
    2018](http://ai.stanford.edu/~cbfinn/_files/dissertation.pdf)).

-   "Meta-learning suggests framing the learning problem at two levels. The
    first is quick acquisition of knowledge within each separate task presented.
    This process is guided by the second, which involves slower extraction of
    information learned across all the tasks."

---

A. Santoro, S. Bartunov, M. Botvinick, D. Wierstra, T. Lillicrap (2016) .
[Meta-Learning with Memory-Augmented Neural
Networks](http://proceedings.mlr.press/v48/santoro16.pdf). *ICML 2016*.

-   Demonstrates that a memory-augmented neural network, based on a Neural
    Turing Machine ([Graves et al., 2014](https://arxiv.org/abs/1410.5401)) and
    Memory Networks ([Weston et al., 2014](https://arxiv.org/abs/1410.3916)) can
    assimilate new data fast and use it to make successful predictions at
    inference after "seeing" only a few examples. (Why one-shot learning:
    machine learning models have to inefficiently relearn the parameters to take
    into account the new information to avoid failing at inference.)

-   "...meta-learning generally refers to a scenario in which an agent learns at
    two levels, each associated with different time scales. Rapid learning
    occurs within a task, for example, when learning to accurately classify
    within a particular dataset. This learning is guided by knowledge accrued
    more gradually across tasks, which captures the way in which task structure
    varies across target domains."

---

D. Samuel, A. Ganeshan, J. Naradowsky (2020). [Meta-learning Extractors for
Music Source Separation](https://arxiv.org/abs/2002.07016). *arXiv:2002.07016*

-   Introduces Meta-TasNet - a waveform-to-waveform meta-learning audio
    separation system adapted from Conv-TasNet ([Luo & Mesgarani,
    2018](https://arxiv.org/abs/1809.07454v3)) for generating
    instrument-specific separation models, each of which is trained separately
    via generator network for parameter prediction (on the MUSDB18 [(Rafii et
    al., 2017)](https://sigsep.github.io/datasets/musdb.html) dataset) ("...a
    generator network "supervises" the training of the individual extractors by
    generating some of their parameters directly").

-   Instead of using a single 1D convolutional layer encoder, Meta-TasNet uses a
    K number of such layers (multiple kernels) to "capture a wider frequency
    range with more fidelity", as well as "features from the classical STFT
    spectrogram of the input mixture, normalizing it, and projecting it down
    with one linear transformation (as a learnable replacement for a mel
    filter)".

-   "A major design choice in music source separation models is whether to (1)
    train a separate model for each instrument \[[Stoter et al.,
    2019](https://www.researchgate.net/publication/335688695_Open-Unmix_-_A_Reference_Implementation_for_Music_Source_Separation)\],
    (2) to use a single class-conditional model, or (3) to use an instrument
    agnostic approach \[[Takahashi et al.,
    2019](https://arxiv.org/abs/1904.03065)\]. Our approach aims to combine the
    advantages of the first two; the high-precision of independent models, with
    improved optimization via parameter sharing in single models. It is also an
    effort to incorporate prior source knowledge into TasNet-type models."

-   Claims the model is "more effective than the models trained independently or
    in a multi-task setting, and achieve performance comparable with
    state-of-the-art methods... In comparison to a single multi-task model, our
    models perform better, and are smaller and faster."

---

O. Gul, C. Schlager, G. Todd (2020). [MuML: Musical
Meta-Learning](https://cs330.stanford.edu/projects2020/CS330_Omer_Gul_Collin_Schlager_Graham_Todd.pdf).
*Stanford CS330 Deep Multi-Task and Meta Learning project*.

-   Attempts to explore the application of Model Agnostic Meta-Learning - MAML
    ([Finn et al., 2017](https://arxiv.org/abs/1703.03400)) - on music
    production (music generation with the ability to predict and generate other
    examples of music per genre).

-   Uses architectures with two-layer LSTM ([Hochreiter & Schmidhuber,
    1997](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735))
    RNNs ([S. El Hihi & Y. Bengio,
    1992](https://papers.nips.cc/paper/1102-hierarchical-recurrent-neural-networks-for-long-term-dependencies.pdf))
    and attention-based Transformers ([Vaswani et al.,
    2017](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)), and
    trains the models on the Lakh MIDI ([Raffel,
    2016](https://colinraffel.com/projects/lmd/)), and Maestro ([Hawthorne et
    al., 2018](https://arxiv.org/abs/1810.12247)) datasets.

-   Demonstrates that the Musical MAML model can perform qualitatively better
    compared with the baseline if it is not positively adapting to task
    information because "the meta- learning objective forces the MAML model to
    place additional emphasis on larger structural features such as melodic and
    rhythmic motifs shared across reference snippets and that this increased
    structural competence is what enables it to produce samples that sound
    similar to those belonging to a particular piece. The baseline model, on the
    other hand, cannot adequately capture the necessary musical information in
    the few-shot training regime with a small model possessing limited
    computational capacity."

-   Discovers that "the meta-learning model better matches the underlying style
    of the provided task", but "when evaluated with negative log-likelihood,
    there is no substantial difference between the MAML and baseline models"
    ("Perhaps the simplest explanation is that the negative log likelihood is
    not effectively capturing musical style" because "despite a realized
    improvement in generating music of a target style, the negative log
    likelihood remains unaffected").

---

W. Liang, Z. Liu, C. Liu (2020). [DAWSON: A Domain Adaptive Few Shot Generation
Framework](https://arxiv.org/abs/2001.00576v1). *Stanford CS 236 Project,*
*arXiv:2001.00576v1.*

-   Introduces the Music MATINEE model that can "quickly adapt to new domains
    with only tens of songs from the target domains" and "learn to generate new
    digits with only four samples in the MNIST dataset."

-   Proposes DAWSON ("a Domain Adaptive Few Shot Generation Framework For GANs")
    that addresses the challenge of meta-learning in GANs ([Goodfellow et al.,
    2014](https://arxiv.org/pdf/1406.2661)) when it comes to obtaining gradients
    for the generator "from evaluating it on development sets due to the
    likelihood-free nature of GANs".

-   The new training process "naturally combines the two-step training procedure
    of GANs and the two-step training procedure of meta-learning algorithms".

-   Adapts methods from the Generative Matching Network ([Bartunov & Vetrov,
    2018](http://proceedings.mlr.press/v84/bartunov18a/bartunov18a.pdf)) and
    one-shot generalization systems by [Rezende et al.,
    2016](https://arxiv.org/abs/1603.05106).

---

T. Munkhdalai, H. Yu (2017). [Meta Networks](https://arxiv.org/abs/1703.00837).
ICML 2017, *arXiv:1703.00837v2.*

-   Introduces MetaNet (Meta Networks) for meta-level continual learning by
    allowing neural networks to learn and to generalize a new task or concept
    from a single example (one-shot learning).

-   Bases the work of various research, including *Using fast weights to deblur
    old memories* ([Hinton & Plaut,
    1987](http://www.cs.toronto.edu/~hinton/absps/fastweights87.pdf)).

---

N. Mishra, M. Rohaninejad, X. Chen, P. Abbeel (2017). [A Simple Neural Attentive
Meta-Learner](https://arxiv.org/abs/1707.03141). *ICLR 2018,*
*arXiv:1707.03141v3.*

-   Introduces the SNAIL (Simple Neural AttentIve Learner) algorithm that
    achieves significant performance improvements in supervised and
    reinforcement learning tasks.

-   Uses a combination of of temporal convolutions ([van den Oord et al.,
    2016](https://arxiv.org/abs/1609.03499v2)) (enabling "the meta-learner to
    aggregate contextual information from past experience") and soft attention
    ([Vaswani et al.,
    2017](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf))
    ("which allow it to pinpoint specific pieces of information within that
    context").

-   Formalizes the meta-learning problem as a sequence-to-sequence problem
    ([Sutskever, Vinyals & Le, 2014](https://arxiv.org/abs/1409.3215)).

---

K. Hsu, S. Levine, C. Finn (2019. [Unsupervised Learning via
Meta-Learning](https://arxiv.org/abs/1810.02334). *ICLR 2019, arXiv:1810.02334*.

-   Presents with a "unsupervised meta-learning approach acquires a learning
    algorithm without any labeled data that is applicable to a wide range of
    downstream classification tasks, improving upon the embedding learned by
    four prior unsupervised learning methods."

    "A central goal of unsupervised learning is to acquire representations from
    unlabelled data or experience that can be used for more effective learning
    of downstream tasks from modest amounts of labeled data."

    (More on unsupervised learning: [Hinton & Sejnowski,
    1999](https://mitpress.mit.edu/books/unsupervised-learning); [Hastie et al.,
    2008](https://link.springer.com/chapter/10.1007/978-0-387-84858-7_14);
    [Stanford\'s Statistical Learning slides (Hastie,
    2020)](http://web.stanford.edu/~hastie/MOOC-Slides/unsupervised.pdf).)

---

T. Nguyen, Z. Chen., J. Lee (2020) [Dataset Meta-Learning from Kernel
Ridge-Regression](https://arxiv.org/abs/2011.00050). *arXiv:2011.00050*.

-   Introduces "the novel concept of ε- approximation of datasets, obtaining
    datasets which are much smaller than or are significant corruptions of the
    original training data while maintaining similar model performance. We
    introduce a meta-learning algorithm called Kernel Inducing Points (KIP) for
    obtaining such remarkable datasets, inspired by the recent developments in
    the correspondence between infinitely-wide neural networks and kernel
    ridge-regression (KRR). For KRR tasks, we demonstrate that KIP can compress
    datasets by one or two orders of magnitude, significantly improving previous
    dataset distillation and subset selection methods while obtaining state of
    the art results for MNIST and CIFAR-10 classification."

-   Shows that KIP-learned datasets "are transferable to the training of
    finite-width neural networks even beyond the lazy-training regime, which
    leads to state of the art results for neural network dataset distillation
    with potential applications to privacy-preservation."

---

T. Hospedales, A. Antoniou, P. Micaelli, A. Storkey (2020). [Meta-Learning in
Neural Networks: A Survey](https://arxiv.org/abs/2004.05439).

-   Provides an overview of the meta-learning landscape: definitions, relation
    to transfer learning and hyperparameter optimization, a new comprehensive
    taxonomy, promising applications and successes (such as few-shot learning
    and reinforcement learning), and outstanding challenges and areas for future
    research.

---

O. Vinyals. [Model vs Optimization Meta
Learning](https://drive.google.com/file/d/15fvtfWNw2T4DcU7Kw1cfT9ZjJTwAFGOg/view).
*NeurIPS 2017 Meta-Learning Symposium*.

-   Outlines the definitions of meta-learning, draws contrasts with supervised
    learning, presents the meta-learning taxonomy (adapted from [Finn et al.,
    2017](https://arxiv.org/abs/1703.03400): model-based meta-learning,
    metric-based meta-learning, and optimization-based meta-learning), and
    discusses future work as of 2017.

---

C. Finn (2018). [Learning to learn with
gradients](http://ai.stanford.edu/~cbfinn/_files/dissertation.pdf). *Thesis*.

-   Related to MAML Model-Agnostic Meta-Learning) ([Finn et al.,
    2017](https://arxiv.org/abs/1703.03400)) (MAML directly optimises
    performance with respect to a model's initial parameters, such that the
    model can adapt to a new similar task faster.).

-   "To study the problem of learning to learn, we first develop a clear and
    formal definition of the meta-learning problem, its terminology, and
    desirable properties of meta-learning algorithms. Building upon these
    foundations, we present a class of model-agnostic meta-learning methods that
    embed gradient-based optimization into the learner. Unlike prior approaches
    to learning to learn, this class of methods focus on acquiring a
    transferable representation rather than a good learning rule. As a result,
    these methods inherit a number of desirable properties from using a fixed
    optimization as the learning rule, while still maintaining full
    expressivity, since the learned representations can control the update
    rule."

---

A. Raghu, S. Bengio, O. Vinyals (2020). [Rapid Learning or Feature Reuse?
Towards Understanding the Effectiveness of
MAML](https://arxiv.org/abs/1909.09157). *ICLR 2020,* *arXiv:1909.09157*.

-   Investigates whether the effectiveness of the MAML (Meta-Agnostic
    Meta-Learning) algorithm ([Finn et al.,
    2017](https://arxiv.org/abs/1703.03400)) is due to the meta-initialization
    being primed for faster learning or due to feature reuse, with the
    meta-initialization already containing high quality features.

-   Introduces the ANIL (Almost No Inner Loop) algorithm - a simpler version of
    MAML, which matches MAML's performance on benchmark few-shot image
    classification and reinforcement learning, while offering computational
    improvements.

---

OpenAI (A. Nichol, J. Achiam, J. Schulman (2018)). [First-Order Meta-Learning
Algorithms](https://arxiv.org/abs/1803.02999). *arXiv:1803.02999*.

-   Provides an analysis of a family of algorithms that meta-learn parameter
    initialization, including MAML (and first-order MAML (FOMAML) that ignores
    second-derivative terms ([Finn et al.,
    2017](https://arxiv.org/abs/1703.03400))).

-   Introduces the Reptile algorithm - extending FOMAML - that doesn't require a
    training-test split for each task.

    Demonstrates that the Reptile model can still learn weights such that
    running gradient descent on similar tasks makes progress fast; provides
    insights for how to best implement the algorithms.

---

A. Antoniou, H. Edwards, A. Storkey (2018). [How to Train Your
MAML](https://arxiv.org/abs/1810.09502). *ICLR 2019*.

-   Presents a variant of MAML (Meta-Agnostic Meta-Learning) ([Finn et al.,
    2017](https://arxiv.org/abs/1703.03400)) that "offers the flexibility of
    MAML along with many improvements, such as robust and stable training,
    automatic learning of the inner loop hyperparameters, greatly improved
    computational efficiency both during inference and training and
    significantly improved generalization performance."

-   Claims that "MAML suffers from a variety of problems which: 1) cause
    instability during training, 2) restrict the model's gener- alization
    performance, 3) reduce the framework's flexibility, 4) increase the system's
    computational overhead and 5) require that the model goes through a costly
    (in terms of time and computation needed) hyperparameter tuning before it
    can work robustly on a new task."

---

T. Wolf, J. Chaumond, C. Delangue (2018). [Meta-Learning a Dynamical Language
Model](https://arxiv.org/abs/1803.10631). *ICLR 2018*.

-   "In this work, we study the possibility of combining short-term
    representations stored in hidden states with medium term representations
    encoded in a set of dynamical weights of the language model."

-   "...extends a series of recent experiments on networks with dynamically
    evolving weights \[[Ba, Hinton, Mnih et al.,
    2016](https://arxiv.org/abs/1610.06258); [HyperNetworks (Ha, Dai & Q. Le,
    2016)](https://arxiv.org/abs/1609.09106); [Krause et al.,
    2017](https://arxiv.org/abs/1709.07432)\] which shows improvements in
    sequential prediction tasks... formulating the task as a hierarchical online
    meta-learning task".

---

C. I. Winata, S. Cahyawijaya, Z. Lin, Z. Liu, P. Xu, P. Fung (2020).
[Meta-Transfer Learning for Code-Switched Speech
Recognition](https://arxiv.org/abs/2004.14228). *ACL 2020*.

-   Proposes a meta-transfer learning method for speech recognition that extends
    MAML (Meta-Agnostic Meta-Learning) ([Finn et al.,
    2017](https://arxiv.org/abs/1703.03400)) "to not only train with monolingual
    source language resources but also optimize the update on the code-
    switching data".

-   The model "learns to recognize individual languages, and transfer them so as
    to better recognize mixed-language speech by conditioning the optimization
    on the code-switching data. Based on experimental results, our model
    outperforms existing baselines on speech recognition and language modeling
    tasks, and is faster to converge."

---

J. Gu, Y. Wang, Y. Chen, K. Cho, V. O.K. Li (2018). [Meta-Learning for
Low-Resource Neural Machine Translation](https://arxiv.org/abs/1808.08437)).
*EMNLP 2018*.

-   Proposes a meta-learning algorithm for low-resource neural machine
    translation (NMT).

-   Extends MAML (Meta-Agnostic Meta-Learning) ([Finn et al.,
    2017](https://arxiv.org/abs/1703.03400)): the algorithm helps find "the
    initialization of model parameters that facilitate fast adaptation for a new
    language pair with a minimal amount of training examples".

-   "...vanilla NMT often lags behind conventional machine translation systems,
    such as statistical phrase-based translation systems... for low-resource
    language pairs."

---

M. Yin, G. Tucker, M. Zhou, S. Levine, C. Finn (2019). [Meta-Learning without
Memorization](https://arxiv.org/abs/1912.03820). *ICLR 2020, arXiv:1912.03820.*

-   Proposes to use meta-regularization to reduce task-overfitting in
    meta-learning.

-   "... current methods require careful design of the meta-training tasks to
    prevent a subtle form of task overfitting, distinct from standard
    overfitting in supervised learning. If the task can be accurately inferred
    from the test input alone, then the task training data can be ignored while
    still achieving low meta-training loss. In effect, the model will collapse
    to one that makes zero-shot decisions."

-   "...based on maximizing the mutual information between the task-training
    data and task-specific parameters... by drawing the meta-parameter... from a
    Gaussian distribution...." ([Pan et al.,
    2020](https://cs330.stanford.edu/projects2020/CS330_Edwin_Pan_Pankaj_Rajak_Shubham_Shrivastava.pdf)).

---

E. Pan, P. Rajak, S. Shrivastava. [Meta-Regularization by Enforcing
Mutual-Exclusiveness](https://cs330.stanford.edu/projects2020/CS330_Edwin_Pan_Pankaj_Rajak_Shubham_Shrivastava.pdf).
*Stanford CS330 Deep Multi-Task and Meta Learning project*.

-   "In the case of optimization based models, we regularize the model by
    maximizing the Euclidean distance between the task-specific parameters
    itself."

-   "In all meta-learning models, each task consists of training and test data,
    where the objective of the model is to quickly converge on a solution for
    the task using task-training data only. However, during training if models
    completely ignore the task-training data while learning about the tasks, it
    leads to task-overfitting. In essence, the model has memorized the
    underlying functional form of all the training tasks in its weight vector,
    and thus does not know how to process the training data of new tasks at
    meta-test time."
