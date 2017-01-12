# haiku_rnn

 This model is based upon a Tensorflow version of Andrej Karpathy's char-rnn that was developed by [Sherjil Ozair](https://github.com/sherjilozair/char-rnn-tensorflow). It has been modified to work with UTF-16 files that better account for all of the characters of Japanese.

 Data included are the haiku of [Matsuo Basho](https://en.wikipedia.org/wiki/Matsuo_Bash%C5%8D) and an incomplete set of the haiku of [Kobayashi Issa](https://en.wikipedia.org/wiki/Kobayashi_Issa). Both were members of the "Great Four" haiku masters in Japan. Issa, in particular, was especially prolific, composing over 20,000 haiku in his lifetime.

 The Basho haiku are cleaned, with one haiku per line. However, Basho composed fewer than 1,000 haiku. The Issa haiku data comes from the website of [Ichiro Kobayashi](http://www.janis.or.jp/users/kyodoshi/issaku.htm), of the Nagano Regional History Study Group. This data is in the process of being cleaned and split onto three lines each. In its current state, the structure of the website from which it was scraped is reflected in the predictions (as would be expected).

 The model can be trained using `train.py` and sampled from using `sample.py`.

**Please contact [Henry Wolf](mailto:aenrichus@gmail.com) if you have access to a larger corpus of East Asian poetry upon which this could be tested.**
 
