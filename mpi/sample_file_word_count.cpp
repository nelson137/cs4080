#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <cmath>

#include <chrono>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/lexical_cast.hpp>

#include <mpi.h>

#define GJS_DEBUG_PRINT 0

namespace scottgs {

typedef std::map<int,std::string> book_type;

enum MPI_TAGS {TERMINATE = 0, PROCESS = 1};

const int LINE_MESSAGE_SIZE = 1000;
const int RESULT_MESSAGE_SIZE = 1500;

// Function Prototypes
scottgs::book_type parseFileToBook(const std::string& filename, std::string& bookName);
std::map<std::string,int> bookCounter(const book_type& book); // coordinating function
void verseCounter();	// worker function
std::string workerCountWords(const std::string& line, int rank);

void mergeBookCounts(std::map<std::string,int>& bookCounter, const std::string result);

}; // end of scottgs namespace

int main(int argc, char * argv[])
{

	// Initialize MPI
	// Find out my identity in the default communicator
	MPI_Init(&argc, &argv);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	// Logic branch
	if (rank == 0) 
	{
		// ++++++++++++++++++++++++++++++
		// Master process
		// ++++++++++++++++++++++++++++++

		// Validate
		if (argc < 2)
		{
			std::cout << "Usage: " << argv[0] << " <text_file>" << std::endl;
			return 1;
		}

		const std::string filename(argv[1]);	
			
		// We will attempt to guess the book name
		std::string bookName;
		
		// Parse the file
		scottgs::book_type book( scottgs::parseFileToBook(filename, bookName) );
	
		// I have parsed the entire text file into book
		// Start some MPI Stuff
		std::cout << "I have prepped " << book.size() << " lines to be counted." << std::endl;
 
		// Invoke the master, map to workers
                std::chrono::high_resolution_clock c;
                std::chrono::high_resolution_clock::time_point start = c.now();
		std::map<std::string,int> bookCounterResults(scottgs::bookCounter(book));
		
		// Barrier to get all the desired output together
		// Warning, if threads do not call Barrier, then 
		// the MPI_Barrier busy waits
#if GJS_DEBUG_PRINT			
		std::cout << "main, barrier reached, waiting on workers" << std::endl;
#endif		  
		MPI_Barrier(MPI_COMM_WORLD);
		std::chrono::high_resolution_clock::time_point stop = c.now();
		const double t = (double) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000000.0;
		
		// Dump the results
		std::cout << "Global Word Count, in " << t << " s" << std::endl
			  << "===================================" << std::endl;  
		for (std::map<std::string,int>::const_iterator t=bookCounterResults.begin();t!=bookCounterResults.end();++t)
		{
			const std::string term(t->first);
			const int count = t->second;
			std::cout << "Term (" << term << ") occurs " << count << " times in "<< bookName << std::endl;
		}
		std::cout << "==================" << std::endl
			  << "Analysis Completed" << std::endl;
	}
	else 
	{	
		// ++++++++++++++++++++++++++++++
		// Workers
		// ++++++++++++++++++++++++++++++
		scottgs::verseCounter();
		MPI_Barrier(MPI_COMM_WORLD); // this is linked to the above barrier
	}

	// Shut down MPI
	MPI_Finalize();
	
	return 0;	
} // END of main

scottgs::book_type scottgs::parseFileToBook(const std::string& filename, std::string &bookName)
{
	// Hold my simple chapter-verse index pointing at the line of text
	scottgs::book_type book;

	// Do the basic IO prep
	// line with hold the read-in file line
	// assume filename is a std::string holding the file to process
	std::string line;
	std::ifstream datafile(filename.c_str()); // open the file for input

        std::chrono::high_resolution_clock c;

	if (datafile.is_open()) // test for open failure
	{

		unsigned int linesRead = 0;
				
		// Process file into memory
        	std::chrono::high_resolution_clock::time_point start = c.now();
		while ( datafile.good() ) // read while the stream is good for reading
		{
	
			// Lines are typically formatted as chapter:verse: Text
			std::getline(datafile,line); // see: http://www.cplusplus.com/reference/string/getline/
		
#if GJS_DEBUG_PRINT			
			std::cout << "READ: {"<< line << "}" << std::endl;
#endif			
			if (line.empty())
				continue; // do not try to parse an empty line
		
			// Break / tokenize / split the line string into a 
			// vector of strings using boost and the ':'
			std::vector<std::string> chapterVerseStory;
			boost::split(chapterVerseStory, line, boost::is_any_of(":"));
	
			if (chapterVerseStory.size() < 3)
			{
				if (chapterVerseStory.size() == 1  && bookName.empty())
				{
					bookName += boost::algorithm::trim_copy(chapterVerseStory[0]);
					std::cout << "BookName? " << bookName << std::endl;
				}
				else if (chapterVerseStory.size() == 1  && !bookName.empty())
				{
					bookName += " " + boost::algorithm::trim_copy(chapterVerseStory[0]);
					std::cout << "BookName? " << bookName << std::endl;
				}

				else
				{
					std::cout << "Non-verse line: (" << line << ")" << std::endl;
				}
			
				// No matter what, we continue to next iteration because this is a non-verse
				continue;
			}
			// Trim
			const std::string chapter(boost::algorithm::trim_copy(chapterVerseStory[0]));
			const std::string verse = boost::algorithm::trim_copy(chapterVerseStory[1]);
			const std::string text = boost::algorithm::trim_copy(chapterVerseStory[2]);
			// Create a line index	
			int chapterVerseIndex = boost::lexical_cast<int>(chapter) * 1000 +  boost::lexical_cast<int>(verse);

#if GJS_DEBUG_PRINT			
			std::cout << "Read [" << chapter << ":" << verse << "], index = " << chapterVerseIndex << std::endl;			
#endif
			if (book.find(chapterVerseIndex) != book.end())
			{
				std::cerr << "Problem with reading and parsing line {" << line << "}" << std::endl;
				std::cerr << "Parsed [" << chapter << ":" << verse << "], index = " << chapterVerseIndex << std::endl;
				std::cerr << "However, book currently has (" << book[chapterVerseIndex] << ") at index = " << chapterVerseIndex << std::endl;
				exit(1);
			}
			book[chapterVerseIndex] = text;
		
			++linesRead;
		}
		datafile.close();

		// Log the file load
		double sumTime = 0.0;
		std::chrono::high_resolution_clock::time_point stop = c.now();
		double splitTime = (double) std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000000. ;
		std::cout << "Read (" << linesRead << ") from (" << filename << ") in " << splitTime << " s " << std::endl;

	}
	else
	{
		std::stringstream msg;
		msg << "Unable to open file '" << filename << "'";
		throw std::runtime_error(msg.str());
	}

	return book;
}

std::map<std::string,int> scottgs::bookCounter(const scottgs::book_type& book)
{
	// =========================
	// Master (thread 0) 
	// =========================
	int threadCount;
	MPI_Comm_size(MPI_COMM_WORLD, &threadCount);
	std::cout << "scottgs::bookCounter, " << (threadCount-1) << " workers are available to process " << book.size() << " verses from book" << std::endl;
	
	
	std::map<std::string,int> bookCounterResult;
	
	
	scottgs::book_type::const_iterator verse = book.begin();
	
	// Start with 1, because the master is =0
	for (int rank = 1; rank < threadCount && verse!=book.end(); ++rank, ++verse) 
	{
		// work tag / index
		int index = verse->first;
		
		const std::string line(verse->second);
		const size_t length = line.size();
		char msg[scottgs::LINE_MESSAGE_SIZE];
		line.copy(msg,length);
		msg[length] = '\0';

		MPI_Send(msg,           /* message buffer */
		     scottgs::LINE_MESSAGE_SIZE,            /* buffer size */
		     MPI_CHAR,          /* data item is an integer */
		     rank,              /* destination process rank */
		     index,  		/* user chosen message tag */
		     MPI_COMM_WORLD);   /* default communicator */
  	
  		
	}
	
	
	// Loop through verses until there is 
	// no more work to be done
	for ( ;verse!=book.end(); ++verse) 
	{

		// Receive results from a worker
	     	char resultMsg[scottgs::RESULT_MESSAGE_SIZE];
  		MPI_Status status;

		// Receive a message from the worker
		MPI_Recv(resultMsg, 			/* message buffer */
			scottgs::RESULT_MESSAGE_SIZE, 	/* buffer size */
			MPI_CHAR, 		/* data item is an integer */
			MPI_ANY_SOURCE,  	/* Recieve from thread */
			MPI_ANY_TAG,		/* tag */
		     	MPI_COMM_WORLD, 	/* default communicator */
		     	&status);
		     	
		const int incomingIndex = status.MPI_TAG;
		const int sourceCaught = status.MPI_SOURCE;
		// Convert the message into a string for parse work
		std::string returnLine(resultMsg);
#if GJS_DEBUG_PRINT			
		std::cout << "scottgs::bookCounter, " << (threadCount-1) << ", recieved (" << returnLine 
			<< "), status = " << incomingIndex << ", from " << sourceCaught << std::endl;
#endif		  

		// Merge the results from this thread to the global results
		scottgs::mergeBookCounts(bookCounterResult, returnLine);


		// work tag / index
		int outgoingIndex = verse->first;
		
		const std::string line(verse->second);
		const size_t length = line.size();
		char msg[scottgs::LINE_MESSAGE_SIZE];
		line.copy(msg,length);
		msg[length] = '\0';

		MPI_Send(msg,		/* message buffer */
		     scottgs::LINE_MESSAGE_SIZE,            /* buffer size */
		     MPI_CHAR,		/* data item is an integer */
		     sourceCaught,	/* destination process rank */
		     outgoingIndex,	/* user chosen message tag */
		     MPI_COMM_WORLD);	/* default communicator */
  	
	}
	
	
	
	// There's no more work to be done, so receive all the outstanding
	// results from the workers
	for (int rank = 1; rank < threadCount; ++rank) 
	{
	     		
	     	char resultMsg[scottgs::RESULT_MESSAGE_SIZE];
  		MPI_Status status;

		// Receive a message from the worker
		MPI_Recv(resultMsg, 			/* message buffer */
			scottgs::RESULT_MESSAGE_SIZE, 	/* buffer size */
			MPI_CHAR, 		/* data item is an integer */
			rank,  			/* Receive from master */
			MPI_ANY_TAG,		
		     	MPI_COMM_WORLD, 	/* default communicator */
		     	&status);
		     	
		const int index = status.MPI_TAG;
		// Convert the message into a string for parse work
		std::string line(resultMsg);
#if GJS_DEBUG_PRINT			
		std::cout << "scottgs::bookCounter, " << (threadCount-1) << ", recieved (" << line << "), status = " << index << std::endl;
#endif		  

		// Merge the results from this thread to the global results
		scottgs::mergeBookCounts(bookCounterResult, line);
	}


	// Tell all the workers to exit by sending an empty message with the
	// TERMINATE tag
	for (int rank = 1; rank < threadCount; ++rank) {
		MPI_Send(0, 0, MPI_INT, rank, scottgs::TERMINATE, MPI_COMM_WORLD);
	}
	
	std::cout << "scottgs::bookCounter, completed processing book" << std::endl;


	return bookCounterResult;
}

void scottgs::verseCounter()
{
	// =========================
	// worker function
	// =========================
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	std::cout << "scottgs::verseCounter["<< rank << "], waiting for work" << std::endl;

	char msg[scottgs::LINE_MESSAGE_SIZE];
  	MPI_Status status;

	while (1) 
	{
		// Receive a message from the master
		MPI_Recv(msg, 			/* message buffer */
			scottgs::LINE_MESSAGE_SIZE, 	/* buffer size */
			MPI_CHAR, 		/* data item is an integer */
			0,  			/* Receive from master */
			MPI_ANY_TAG,		
		     	MPI_COMM_WORLD, 	/* default communicator */
		     	&status);

		// Check if we have been terminated by the master
		// exit from the worker loop
		if (status.MPI_TAG == scottgs::TERMINATE) 
		{
			std::cout << "scottgs::verseCounter["<< rank << "], recieved terminate signal" << std::endl;
			return;
		}
		
		const int index = status.MPI_TAG;
		
		// Convert the message into a string for parse work
		std::string line(msg);
#if GJS_DEBUG_PRINT			
		std::cout << "scottgs::verseCounter[" << rank << "], recieved (" << line << "), status = " << index << std::endl;
#endif		  

		// We have now built up a term=count comma separated list
		// We can send the result back
		const std::string resultLine = scottgs::workerCountWords(line, rank);
		const size_t length = resultLine.size();
		char resultMsg[scottgs::RESULT_MESSAGE_SIZE];
		resultLine.copy(resultMsg,length);
		resultMsg[length] = '\0';
#if GJS_DEBUG_PRINT			
		std::cout << "scottgs::verseCounter[" << rank << "], result list (" << resultLine << ")" << std::endl;
#endif		  
		MPI_Send(resultMsg,           /* message buffer */
		     scottgs::RESULT_MESSAGE_SIZE,            /* buffer size */
		     MPI_CHAR,          	/* data item is an integer */
		     0,              		/* destination process rank, the master */
		     index,  			/* user chosen message tag */
		     MPI_COMM_WORLD);   	/* default communicator */
		
	} // end of an infinite loop
	return;
}

std::string scottgs::workerCountWords(const std::string& line, int rank)
{

	// Split on any punctuation or space
	// build into maps with a count
	std::map<std::string,int> wordCounts;
	std::vector<std::string> words;
	boost::split(words, line, boost::is_any_of(": ,.;!?)("));
#if GJS_DEBUG_PRINT			
	std::cout << "scottgs::verseCounter[" << rank << "], found token count = " << words.size() << std::endl;
#endif		  

	for(std::vector<std::string>::const_iterator w=words.begin();w!=words.end();++w)
	{
		const std::string term(boost::algorithm::to_lower_copy(*w));
		
		if (boost::algorithm::trim_copy(term).empty()) continue;	// ignore empty strings
			
		if (wordCounts.find(term) == wordCounts.end())
		{
			// term not yet counted
			wordCounts[term] = 1;
		}
		else
		{
			// term found
			wordCounts[term] += 1;
		}
	}
#if GJS_DEBUG_PRINT			
	std::cout << "scottgs::verseCounter[" << rank << "], found " << wordCounts.size() << " distinct tokens" << std::endl;
#endif		  
	std::stringstream result;
	for (std::map<std::string,int>::const_iterator t=wordCounts.begin();t!=wordCounts.end();++t)
	{
		const std::string term(t->first);
		const int count = t->second;
#if GJS_DEBUG_PRINT			
		std::cout << "scottgs::verseCounter[" << rank << "], term (" << term << ") count is " << count << std::endl;
#endif		  
		result << term << "="<< count << ",";
	}

	return result.str();
}

void scottgs::mergeBookCounts(std::map<std::string,int>& bookCounter, const std::string result)
{
	// Split on any punctuation or space
	// build into maps with a count
	std::vector<std::string> resultTermCount;
	boost::split(resultTermCount, result, boost::is_any_of(","));
#if GJS_DEBUG_PRINT			
	std::cout << "scottgs::mergeBookCounts, found token count = " << resultTermCount.size() << std::endl;
#endif		  

	// If there is no data, early return
	if (resultTermCount.size() < 1) return;

	for(std::vector<std::string>::const_iterator w=resultTermCount.begin();w!=resultTermCount.end();++w)
	{
		// Parse out the term and the count
		const std::string line(*w);
		std::vector<std::string> singleTermCount;
		boost::split(singleTermCount, line, boost::is_any_of("="));

		if (singleTermCount.size() < 2) continue; // skip if we had a term that was not <term>=<count>

		const std::string term(singleTermCount[0]);
		
		if (boost::algorithm::trim_copy(term).empty()) continue;	// ignore empty strings
			
		if (bookCounter.find(term) == bookCounter.end())
		{
			// term not yet counted
			bookCounter[term] = boost::lexical_cast<int>(singleTermCount[1]);
		}
		else
		{
			// term found
			bookCounter[term] += boost::lexical_cast<int>(singleTermCount[1]);
		}
	}
}
