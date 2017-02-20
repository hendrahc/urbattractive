class PagesController < ApplicationController
  def index
      @title = "Home"
      if params[:user].nil? && params[:campaign].nil? && params[:mw].nil?
        session[:campaign] = 99
        session[:userid] = rand(1..1000)
        session[:locs] = CampaignSet.find(session[:campaign]).locs
        session[:scale] = CampaignSet.find(session[:campaign]).scale
        redirect_to newuser_path
      elsif 
        session[:campaign] = 99  # session[:campaign] = params[:campaign]
        session[:userid] = params[:user].to_s
        @finalstring = params[:mw] += params[:user] += "8c25a8f5e42c00a6f814f45ac764084f4b20a5c476be47ac2a674b82d0ba541f"
        session[:vcode] = Digest::SHA2.hexdigest(@finalstring)
        session[:vcode] = "mw-" + session[:vcode].to_s
        session[:locs] = CampaignSet.find(session[:campaign]).locs
        session[:scale] = CampaignSet.find(session[:campaign]).scale
        session[:microworkers] = '1'
        session[:content] = nil
      end
  end

  def intro
      @title = "Instructions"
  end

  def help
      @title = "Help"
  end

  def about
      @title = "About"
  end

  def contact
  end

  def end
    session[:userid] = nil
    session[:campaign] = nil
    session[:locs] = nil
    session[:img_num] = nil
    session[:training] = nil
    session[:scale] = nil
    session[:microworkers] = nil
    session[:content] = nil
  end

  def training
    @title = "Training"
    @score = Score.new
    if session[:training] == nil
      session[:training] = 'start'
    elsif session[:training] == 'start'
      session[:training] = 'finish'
    elsif session[:training] == 'finish'
        redirect_to ready_path
    end
  end

  def trainreco
    @title = "Training"
  end

  def trainaest
    @title = "Training"
  end

   def trainend
    @title = "Training"
  end

  def ready_part1
    @title = "Begin Test"
     #randomize images
     session[:locs] = session[:locs].split(" ")
     session[:images] = Image.where(Image.arel_table[:loc_id].in(session[:locs])).ids
     session[:img_num] = '0';
     session[:part] = 1
  end

  def ready_part2
    @title = "Begin Test - Part 2"
    session[:loc_num] = '0';
    session[:part] = 2
  end

  def totraining
    @title = "Training"
    session[:training] = nil
  end

  def submit_golden
    if session[:part] == 1
        redirect_to show_part1_path
    else
        redirect_to show_part2_path
    end
  end

  def endsubtask
  end

  def intro0
  end

end
