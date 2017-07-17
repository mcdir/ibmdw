
import feedparser
import pickle
import os
from example_capture import CaptureFeeds


class ReadReutersData:

    def __init__(self):
        for (i, url) in enumerate(self.rss_feeds_list()):
            self.capture_as_pickled_feed(url.strip(), i)

    def rss_feeds_list(self):
        f = open('feeds_list.txt', 'r')
        list = f.readlines()
        f.close
        return list

    def reuters_metadata_dir(self):
        return os.path.expanduser('~/nltk_data/corpora/reuters/')

    def read_reuters_metadata(self):
        cats_file = "%s%s" % (self.reuters_metadata_dir(), 'cats.txt')
        f = open(cats_file, 'r')
        lines = f.readlines()
        f.close()
        return lines

    def capture_as_pickled_feed(self, url, feed_index):
        feed = feedparser.parse(url)
        f = open('data/feed_' + str(feed_index) + '.pkl', 'wb')
        pickle.dump(feed, f)
        f.close()

if __name__ == "__main__":
    cf = ReadReutersData()