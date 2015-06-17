#define BOOST_FILESYSTEM_NO_DEPRECATED

#include<boost/regex.hpp>
#include<boost/filesystem/operations.hpp>
#include<boost/filesystem/path.hpp>
#include<boost/progress.hpp>
#include<iostream>
#include<string>

#include <vector>
#include <cmath>
#include <mpi.h>
#include <fstream>
#include <cstdlib>

namespace fs=boost::filesystem;
void   get_indices( int rank ,int file_count,int size ,int* js, int* je){
  (*js) = 0;
  int residual = file_count % size;
  if( rank < residual ){
    *js = rank*( file_count/size + 1);
    *je = *js + (file_count/size+1) -1;
  }
  else {
    *js = residual*( file_count/size +1 ) + (rank - residual )*(file_count/size);
    *je = *js + file_count/size -1;
  }
  //std::cout << "file_count " << file_count/size << std::endl;
  return;
}
      
int main( int argc, char* argv[] )
{
  int rank, size;
  MPI_File f1;
  MPI_Status status;
  int ngrids;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  
  fs::path full_path( fs::initial_path<fs::path>() );
  std::vector< std::string > all_matching_files;
  boost::regex my_filter("checkpoint.*\\.h5");
  // . wildcard: Matches any single character except \n.
  // * matches the previous element zero or more times.
  // + matches the previous element one or more times.
  
  if ( argc > 1)
    full_path = fs::system_complete( fs::path( argv[2] ) );
  else
    std::cout << "\nusage: ./exe  *.py  Inputdir Outputdir" << std::endl;

  unsigned long file_count = 0;

  if ( !fs::exists( full_path ) )
    {
      std::cout << "\nNot found: " << full_path.string() << std::endl;
      return 1;
    }

  if ( fs::is_directory( full_path ) )
    {
      std::cout << "\nIn directory: "
		<< full_path.string() << "\n";
      fs::directory_iterator end_iter;
      for ( fs::directory_iterator dir_itr( full_path );
	    dir_itr !=end_iter;
	    ++dir_itr )
	{
	  if ( !fs::is_regular_file( dir_itr->status() ) ) continue;
	  boost::smatch matches;
	  if( !boost::regex_match( dir_itr->path().filename().string() , matches, my_filter ) )  continue;
	  ++file_count;
	  all_matching_files.push_back( dir_itr->path().filename().string() );
	  
	}
    } else {
    std::cout <<"\nFound: "<< full_path.string() << "\n";
  }
  std::cout << file_count << "files\n" << std::endl;
  
  int i, js, je;
  get_indices( rank ,file_count, size , &js, &je);
  if( !system(NULL) )  exit( EXIT_FAILURE);

  std::string command;
  std::vector<std::string> params(argv,argv+argc);
  std::cout << "rank " <<rank << " js, je "<< js <<" "<< je << std::endl;
  char command_char[1024];
  //std::cout << "file_count, rank, size" << file_count <<" "<< rank <<" "<< size << std::endl;
  for( i=js; i<=je; i++){
    command = "python " + params[1] + " "+ full_path.string() + all_matching_files[i] + " " + params[3] + "/" +  all_matching_files[i]+".png";
    //std::cout<< command <<std::endl;
    strcpy(command_char, command.c_str());
    system(command_char);
  }
  
  MPI_Finalize();
}

  
	  
	
