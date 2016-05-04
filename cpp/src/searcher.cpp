/**
 * @file searcher.cpp
 * @author Sean Massung
 * @author Chase Geigle
 */

#include <algorithm>
#include <json/json.h>

#include "meta/corpus/document.h"
#include "meta/index/ranker/ranker.h"
#include "meta/index/ranker/ranker_factory.h"
#include "meta/index/ranker/okapi_bm25.h"
#include "meta/index/ranker/pivoted_length.h"
#include "meta/index/ranker/dirichlet_prior.h"
#include "meta/index/ranker/jelinek_mercer.h"
#include "meta/index/ranker/absolute_discount.h"
#include "meta/logging/logger.h"
#include "meta/util/time.h"
#include "meta/io/filesystem.h"
#include "searcher.h"

searcher::searcher(std::shared_ptr<meta::index::inverted_index> idx)
    : idx_(std::move(idx))
{
}

std::string searcher::search(const std::string& request)
{
    using namespace meta;

    Json::Value json_request;
    Json::Reader reader;
    reader.parse(request.c_str(), json_request);
    auto ranker_method = json_request["ranker"].asString();
    auto query_text = json_request["query"].asString();

    LOG(info) << "Running query using " << ranker_method << ": \""
              << query_text.substr(0, 40) << "...\"" << ENDLG;

    Json::Value json_ret{Json::objectValue};
    meta::corpus::document query;
    query.content(query_text);

    json_ret["results"] = Json::arrayValue;

    auto config = cpptoml::make_table();
    auto elapsed = meta::common::time(
        [&]()
        {
            std::unique_ptr<meta::index::ranker> ranker;
            try
            {
                                // TODO:
                // 1. front-end needs to send an array of params, figure out coffeescript
                // 2. add conditionals for all given rankers and set params
                // 3. why do doc_name() and doc_path() not work? 
                if(!ranker_method.compare("pivoted-length")){
                    double s=0.25;
                    auto s_string = json_request["s"].asString();
                    s = std::stod(s_string);
                    std::cout << "using this" << s << std::endl;
                    ranker = make_unique<meta::index::pivoted_length>(s);
                }
                else if(!ranker_method.compare("bm25")){
                    double k1, b, k3;
                    auto k1_string = json_request["k1"].asString();
                    auto b_string = json_request["b"].asString();
                    auto k3_string = json_request["k3"].asString();
                    k1 = std::stod(k1_string);
                    b = std::stod(b_string);
                    k3 = std::stod(k3_string);
                    std::cout << "using this" << k1 << b << k3 << std::endl;
                    ranker = make_unique<meta::index::okapi_bm25>(k1,b,k3);
                }
                else if(!ranker_method.compare("dirichlet-prior")){
                    double mu=0.25;
                    auto mu_string = json_request["mu"].asString();
                    mu = std::stod(mu_string);
                    std::cout << "using this" << mu << std::endl;
                    ranker = make_unique<meta::index::dirichlet_prior>(mu);
                }
                else if(!ranker_method.compare("jelinek-mercer")){
                    double lambda=0.25;
                    auto lambda_string = json_request["lambda"].asString();
                    lambda = std::stod(lambda_string);
                    std::cout << "using this" << lambda_string << std::endl;
                    ranker = make_unique<meta::index::jelinek_mercer>(lambda);
                }
                else if(!ranker_method.compare("absolute-discount")){
                    double delta=0.25;
                    auto delta_string = json_request["d"].asString();
                    delta = std::stod(delta_string);
                    std::cout << "using this" << delta_string<< std::endl;
                    ranker = make_unique<meta::index::absolute_discount>(delta); 
                }
            }
            catch (meta::index::ranker_factory::exception& ex)
            {
                LOG(error) << " -> couldn't create ranker, defaulting to bm25"
                           << ENDLG;
                ranker = make_unique<meta::index::okapi_bm25>();
            }

            for (auto& result : ranker->score(*idx_, query, 50))
            {
                Json::Value obj{Json::objectValue};
                obj["score"] = result.score;
                obj["name"] = idx_->doc_name(result.d_id);
                obj["path"] = idx_->doc_path(result.d_id);
                json_ret["results"].append(obj);
            }
        });

    json_ret["elapsed_time"] = static_cast<double>(elapsed.count());

    Json::StyledWriter styled_writer;
    auto json_str = styled_writer.write(json_ret);
    std::cout << json_str;

    LOG(info) << "Done running query. (" << elapsed.count() << "ms)" << ENDLG;
    return json_str;
}
