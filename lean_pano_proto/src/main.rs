mod common;
mod sets;
mod v1;
mod v2;
mod v3;
mod v4;
mod v5;
mod v6;
mod v7;
mod v8;
mod v9;
mod v10;
mod v11;
mod v12;
mod v13;
mod v14;
mod v15;
mod v16;
mod v17;
mod v18;
mod v19;
mod v20;
mod v21;
mod v22;
mod v23;
mod v24;
mod v25;
mod v26;
mod v27;
mod v28;
mod v29;
mod v30;
mod v37;
mod v38;
mod v39;
mod v40;
mod v41;
mod v42;
mod v46;
mod v47;
mod v48;
mod v49;
mod v51;
mod v52;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let version = if args.first().is_some_and(|arg| arg == "v1" || arg == "v2" || arg == "v3" || arg == "v4" || arg == "v5" || arg == "v6" || arg == "v7" || arg == "v8" || arg == "v9" || arg == "v10" || arg == "v11" || arg == "v12" || arg == "v13" || arg == "v14" || arg == "v15" || arg == "v16" || arg == "v17" || arg == "v18" || arg == "v19" || arg == "v20" || arg == "v21" || arg == "v22" || arg == "v23" || arg == "v24" || arg == "v25" || arg == "v26" || arg == "v27" || arg == "v28" || arg == "v29" || arg == "v30" || arg == "v31" || arg == "v37" || arg == "v38" || arg == "v39" || arg == "v40" || arg == "v41" || arg == "v42" || arg == "v46" || arg == "v47" || arg == "v48" || arg == "v49" || arg == "v51" || arg == "v52") {
        args.remove(0)
    } else {
        "v2".to_string()
    };
    match version.as_str() {
        "v1" => v1::run(&args),
        "v3" => v3::run(&args),
        "v4" => v4::run(&args),
        "v5" => v5::run(&args),
        "v6" => v6::run(&args),
        "v7" => v7::run(&args),
        "v8" => v8::run(&args),
        "v9" => v9::run(&args),
        "v10" => v10::run(&args),
        "v11" => v11::run(&args),
        "v12" => v12::run(&args),
        "v13" => v13::run(&args),
        "v14" => v14::run(&args),
        "v15" => v15::run(&args),
        "v16" => v16::run(&args),
        "v17" => v17::run(&args),
        "v18" => v18::run(&args),
        "v19" => v19::run(&args),
        "v20" => v20::run(&args),
        "v21" => v21::run(&args),
        "v22" => v22::run(&args),
        "v23" => v23::run(&args),
        "v24" => v24::run(&args),
        "v25" => v25::run(&args),
        "v26" => v26::run(&args),
        "v27" => v27::run(&args),
        "v28" => v28::run(&args),
        "v29" => v29::run(&args),
        "v30" => v30::run(&args),
        "v37" => v37::run(&args),
        "v38" => v38::run(&args),
        "v39" => v39::run(&args),
        "v40" => v40::run(&args),
        "v41" => v41::run(&args),
        "v42" => v42::run(&args),
        "v46" => v46::run(&args),
        "v47" => v47::run(&args),
        "v48" => v48::run(&args),
        "v49" => v49::run(&args),
        "v51" => v51::run(&args),
        "v52" => v52::run(&args),
        _ => v2::run(&args),
    }
}
