class PagesController < ApplicationController
  def index
      @title = "Home"
      if params[:user].nil? && params[:campaign].nil? && params[:mw].nil?
        redirect_to newuser_path
      else
         redirect_to ready_part1_path
      end


      #elsif
      #  session[:campaign] = 99  # session[:campaign] = params[:campaign]
      #  session[:userid] = params[:user].to_s
      #  @finalstring = params[:mw] += params[:user] += "8c25a8f5e42c00a6f814f45ac764084f4b20a5c476be47ac2a674b82d0ba541f"
      #  session[:vcode] = Digest::SHA2.hexdigest(@finalstring)
      #  session[:vcode] = "mw-" + session[:vcode].to_s
      #  session[:locs] = CampaignSet.find(session[:campaign]).locs
      #  session[:scale] = CampaignSet.find(session[:campaign]).scale
      #  session[:microworkers] = '1'
      #  session[:content] = nil
      #end
  end

  def intro_part1
      @title = "Instructions - Part 1"
  end

  def intro_part2
        @title = "Instructions - Part 2"
    end

  def help
      @title = "Help"
  end

  def about
      @title = "About"
  end

  def contact
    @title = "Contact"
  end

  def end
    @title = "End"
  end

  def golden
    if session[:part] == 1
        @title = "Part 1"
    else
        @title = "Part 2"
     end
  end

  def logout
    @user = User.find(session[:userid])
    @user.end_time = Time.now.strftime("%I:%M:%S %z")
    @user.save

    session[:userid] = nil
    session[:campaign] = nil
    session[:locs] = nil
    session[:img_num] = nil
    session[:training] = nil
    session[:scale] = nil
    session[:microworkers] = nil
    session[:content] = nil

    redirect_to root_path
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
    @title = "Begin Part 1"
  end

  def ready_part2
    @title = "Begin Part 2"
    session[:loc_num] = '0';
    session[:part] = 2
  end

  def totraining_part1
    @title = "Example - Part 1"
    session[:training] = nil
  end

  def training_part1
      @title = "Example - Part 1"
      session[:training] = nil
  end

  def totraining_part2
      @title = "Trial - Part 1"
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
