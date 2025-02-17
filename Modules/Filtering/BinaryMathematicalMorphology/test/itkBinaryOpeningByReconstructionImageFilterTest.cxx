/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkSimpleFilterWatcher.h"

#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryOpeningByReconstructionImageFilter.h"
#include "itkTestingMacros.h"


int
itkBinaryOpeningByReconstructionImageFilterTest(int argc, char * argv[])
{

  if (argc != 6)
  {
    std::cerr << "Missing parameters." << std::endl;
    std::cerr << "Usage: " << itkNameOfTestExecutableMacro(argv);
    std::cerr << " input output conn fg kernelSize" << std::endl;
    return EXIT_FAILURE;
  }

  constexpr int dim = 2;

  using IType = itk::Image<unsigned char, dim>;

  using ReaderType = itk::ImageFileReader<IType>;
  auto reader = ReaderType::New();
  reader->SetFileName(argv[1]);
  reader->Update();

  using KernelType = itk::BinaryBallStructuringElement<bool, dim>;
  KernelType           ball;
  KernelType::SizeType ballSize;
  ballSize.Fill(std::stoi(argv[5]));
  ball.SetRadius(ballSize);
  ball.CreateStructuringElement();

  using I2LType = itk::BinaryOpeningByReconstructionImageFilter<IType, KernelType>;
  auto reconstruction = I2LType::New();
  reconstruction->SetInput(reader->GetOutput());
  reconstruction->SetKernel(ball);
  reconstruction->SetFullyConnected(std::stoi(argv[3]));
  reconstruction->SetForegroundValue(std::stoi(argv[4]));
  //   reconstruction->SetBackgroundValue( std::stoi(argv[6]) );
  itk::SimpleFilterWatcher watcher(reconstruction, "filter");

  using WriterType = itk::ImageFileWriter<IType>;
  auto writer = WriterType::New();
  writer->SetInput(reconstruction->GetOutput());
  writer->SetFileName(argv[2]);

  ITK_TRY_EXPECT_NO_EXCEPTION(writer->Update());


  return EXIT_SUCCESS;
}
