#include <gtest/gtest.h>

#include "ort_core/ort_core.h"
#include "tests/fps_counter.h"
#include "detection_2d_util/detection_2d_util.h"
#include "mono_stereo_depth_anything/depth_anything.hpp"

/**************************
****  ort core test ****
***************************/

using namespace inference_core;
using namespace detection_2d;
using namespace stereo;

std::shared_ptr<BaseMonoStereoModel> CreateModel()
{
  auto engine = CreateOrtInferCore("/workspace/models/depth_anything_v2_vits.onnx");
  auto preprocess_block = CreateCpuDetPreProcess({123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}, true, true);
  auto model            = stereo::CreateDepthAnythingModel(engine, preprocess_block, 518, 518,
                                                         {"images"}, {"depth"});

  return model;
}

std::tuple<cv::Mat> ReadTestImages()
{
  auto image  = cv::imread("/workspace/test_data/left.png");

  return {image};
}

TEST(depth_anything_test, ort_core_correctness)
{
  auto model         = CreateModel();
  auto [image] = ReadTestImages();

  cv::Mat depth;
  model->ComputeDepth(image, depth);

  double minVal, maxVal;
  cv::minMaxLoc(depth, &minVal, &maxVal);
  cv::Mat normalized_disp_pred;
  depth.convertTo(normalized_disp_pred, CV_8UC1, 255.0 / (maxVal - minVal),
                 -minVal * 255.0 / (maxVal - minVal));


  cv::Mat color_normalized_disp_pred;
  cv::applyColorMap(normalized_disp_pred, color_normalized_disp_pred, cv::COLORMAP_JET);
  cv::imwrite("/workspace/test_data/lightstereo_result_color.png", color_normalized_disp_pred);
}

TEST(depth_anything_test, ort_core_speed)
{
  auto model         = CreateModel();
  auto [image] = ReadTestImages();

  FPSCounter fps_counter;
  fps_counter.Start();
  for (int i = 0; i < 200; ++i)
  {
    cv::Mat depth;
    model->ComputeDepth(image, depth);
    fps_counter.Count(1);
    if (i % 20 == 0)
    {
      LOG(WARNING) << "cur fps : " << fps_counter.GetFPS();
    }
  }
}



TEST(depth_anything_test, ort_core_pipeline_correctness)
{
  auto model         = CreateModel();
  model->InitPipeline();
  auto [image] = ReadTestImages();

  auto async_func = [&]() {
    return model->ComputeDepthAsync(image);
  };

  auto thread_fut = std::async(std::launch::async, async_func);

  auto stereo_fut = thread_fut.get();

  CHECK(stereo_fut.valid());

  cv::Mat depth = stereo_fut.get();

  double minVal, maxVal;
  cv::minMaxLoc(depth, &minVal, &maxVal);
  cv::Mat normalized_disp_pred;
  depth.convertTo(normalized_disp_pred, CV_8UC1, 255.0 / (maxVal - minVal),
                 -minVal * 255.0 / (maxVal - minVal));

  cv::Mat color_normalized_disp_pred;
  cv::applyColorMap(normalized_disp_pred, color_normalized_disp_pred, cv::COLORMAP_JET);
  cv::imwrite("/workspace/test_data/lightstereo_result_color.png", color_normalized_disp_pred);
}


TEST(depth_anything_test, ort_core_pipeline_speed)
{
  auto model         = CreateModel();
  model->InitPipeline();
  auto [image] = ReadTestImages();

  deploy_core::BlockQueue<std::shared_ptr<std::future<cv::Mat>>> future_bq(100);

  auto func_push_data = [&]() {
    int index = 0;
    while (index++ < 200)
    {
      auto p_fut = std::make_shared<std::future<cv::Mat>>(
          model->ComputeDepthAsync(image.clone()));
      future_bq.BlockPush(p_fut);
    }
    future_bq.SetNoMoreInput();
  };

  FPSCounter fps_counter;
  auto       func_take_results = [&]() {
    int index = 0;
    fps_counter.Start();
    while (true)
    {
      auto output = future_bq.Take();
      if (!output.has_value())
        break;
      output.value()->get();
      fps_counter.Count(1);
      if (index ++ % 20 == 0) {
        LOG(WARNING) << "average fps: " << fps_counter.GetFPS();
      }
    }
  };

  std::thread t_push(func_push_data);
  std::thread t_take(func_take_results);

  t_push.join();
  model->StopPipeline();
  t_take.join();
  model->ClosePipeline();

  LOG(WARNING) << "average fps: " << fps_counter.GetFPS();
}
