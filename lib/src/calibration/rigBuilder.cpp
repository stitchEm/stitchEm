// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "rigBuilder.hpp"
#include "camera.hpp"
#include "eigengeometry.hpp"

#include "common/angles.hpp"

namespace VideoStitch {
namespace Calibration {

/* returns the sum of weight along the Minimum Spanning Tree */
double RigBuilder::build(std::vector<std::shared_ptr<Camera> >& cameras, const RigGraph& rigGraph,
                         const unsigned int center) {
  RigGraph::EdgeList traversedEdges = rigGraph.bfsTraversal(rigGraph.mst(), center);
  double weights = 0;

  /**Reference frame 'ref' is the node 'center'*/
  for (RigGraph::EdgeList::const_iterator e = traversedEdges.begin(); e != traversedEdges.end(); ++e) {
    std::shared_ptr<Camera> camref = cameras[e->getFirst()];

    /**Retrieve absolute transformation of first node : firstTref*/
    Eigen::Matrix3d R = camref->getRotation();

    /* Multiply with relative transformation of second node : secondTref = secondTfirst * firstTref */
    Eigen::Matrix3d updatedR = (Eigen::Matrix3d)(e->getPayload() * R);

    std::shared_ptr<Camera> camdest = cameras[e->getSecond()];

    /* Update calibration params, keep original variance */
    camdest->setRotationMatrix(updatedR);

    /* Accumulate weights */
    weights += e->getWeight();
  }

  return weights;
}

}  // namespace Calibration
}  // namespace VideoStitch
