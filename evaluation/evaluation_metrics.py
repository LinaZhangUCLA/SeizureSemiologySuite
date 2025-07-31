from bert_score import BERTScorer
from bert_score import score      # has limit of ~ 512 tokens



def bert_score(reference=None, candidate=None):

    # Initialize the BERTScorer object
    scorer = BERTScorer(lang='en')
    
    # Compute the BERTScore
    P, R, F1 = scorer.score(candidate, reference)

    # Print the scores
    print("Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(P.item(), R.item(), F1.item()))




def main():
    
    chat = """BPM (beats per minute) measures the tempo or speed of a song by counting how many beats occur in one minute."""#To find the BPM, you can listen to the song and count the number of beats within a 15-second interval, then multiply that number by four. Musicians and producers often use metronomes or digital audio software to determine or set the exact BPM. Understanding BPM helps match the rhythm of different tracks, especially in dance music, DJing, or composing."""
    claude = """BPM (beats per minute) measures the tempo or speed of a song by counting how many beats occur within one minute of music."""#To find the BPM, you can tap along to the beat of a song for 15 seconds, count the taps, then multiply by 4 to get the full minute count. Most modern music software and apps can automatically detect BPM by analyzing the audio waveform and identifying the rhythmic pulse patterns. A slow ballad might have a BPM around 60-80, while fast dance music can reach 120-140 BPM or higher.RetryClaude can make mistakes. Please double-check responses."""
    
    # Output: "Precision: 0.90, Recall: 0.92, F1: 0.91"
    bert_score(reference=[chat], candidate=[claude])
    

if __name__ == "__main__":
    main()